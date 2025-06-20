from os.path import join
from itertools import product

from nipype import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Function, Split
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.fsl import (
    IsotropicSmooth, Level1Design, FEATModel,
    L2Model, Merge, FLAMEO, FILMGLS, MultipleRegressDesign,
    FSLCommand, Cluster, FLIRT, Info, ImageMaths
    )
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.fsl.maths import MultiImageMaths

from narps_open.utils.configuration import Configuration
from narps_open.pipelines import Pipeline
from narps_open.data.task import TaskInformation
from narps_open.data.participants import get_group
from narps_open.core.common import list_intersection, elements_in_string, clean_list
from narps_open.core.interfaces import InterfaceFactory

# Setup FSL
FSLCommand.set_default_output_type('NIFTI_GZ')

class PipelineTeamO21U(Pipeline):
    """ A class that defines the pipeline of team O21U """

    def __init__(self):
        super().__init__()
        self.fwhm = 5.0
        self.team_id = 'O21U'
        self.contrast_list = ['1', '2']
        self.run_level_contrasts = [
            ('effect_of_gain', 'T', ['gain', 'loss'], [1, 0]),
            ('effect_of_loss', 'T', ['gain', 'loss'], [0, 1])
            ]

    def get_preprocessing(self):
        """ No preprocessing has been done by team O21U """
        return None

    def get_subject_information(event_file):
        """
        Create Bunchs for specifyModel.

        Parameters :
        - event_file : str, file corresponding to the run and the subject to analyze

        Returns :
        - subject_info : list of Bunch for 1st level analysis.
        """
        from numpy import mean
        from nipype.interfaces.base import Bunch

        onsets = []
        durations = []
        amplitudes_trial = []
        amplitudes_gain = []
        amplitudes_loss = []

        with open(event_file, 'rt') as file:
            next(file)  # skip the header

            for line in file:
                info = line.strip().split()
                onsets.append(float(info[0]))
                durations.append(float(info[1]))
                amplitudes_trial.append(1.0)
                amplitudes_gain.append(float(info[2]))
                amplitudes_loss.append(float(info[3]))

        return [
            Bunch(
                conditions = ['trial', 'gain', 'loss'],
                onsets = [onsets] * 3,
                durations = [durations] * 3,
                amplitudes = [amplitudes_trial, amplitudes_gain, amplitudes_loss]
                )
            ]

    def get_confounds_file(filepath, subject_id, run_id):
        """
        Create a tsv file with only desired confounds per subject per run.

        Parameters :
        - filepath : path to the subject confounds file (i.e. one per run)
        - subject_id : subject for whom the 1st level analysis is made
        - run_id: run for which the 1st level analysis is made

        Return :
        - confounds_file : paths to new files containing only desired confounds.
        """
        from os.path import abspath

        from pandas import read_csv, DataFrame
        from numpy import array, transpose

        data_frame = read_csv(filepath, sep = '\t', header=0)
        retained_confounds = DataFrame(transpose(array([
            data_frame['FramewiseDisplacement'], data_frame['X'], data_frame['Y'], data_frame['Z'],
            data_frame['RotX'], data_frame['RotY'], data_frame['RotZ']
        ])))

        # Write confounds to a file
        confounds_file = abspath(f'confounds_file_sub-{subject_id}_run-{run_id}.tsv')
        with open(confounds_file, 'w', encoding = 'utf-8') as writer:
            writer.write(retained_confounds.to_csv(
                sep = '\t', index = False, header = False, na_rep = '0.0'))

        return confounds_file

    def get_run_level_analysis(self):
        """
        Create the run level analysis workflow.

        Returns:
            - run_level : nipype.WorkFlow
        """
        # Create run level analysis workflow and connect its nodes
        run_level = Workflow(
            base_dir = self.directories.working_dir,
            name = 'run_level_analysis'
            )

        # IdentityInterface Node - Iterate on subject and runs
        information_source = Node(IdentityInterface(
            fields = ['subject_id', 'run_id']),
            name = 'information_source')
        information_source.iterables = [
            ('subject_id', self.subject_list),
            ('run_id', self.run_list)
            ]

        # SelectFiles - Get necessary files
        templates = {
            'func' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'),
            'mask' : join('derivatives', 'fmriprep', 'sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz'),
            'events' : join('sub-{subject_id}', 'func',
                'sub-{subject_id}_task-MGT_run-{run_id}_events.tsv')
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory = self.directories.dataset_dir
        run_level.connect(information_source, 'subject_id', select_files, 'subject_id')
        run_level.connect(information_source, 'run_id', select_files, 'run_id')

        # MultiImageMaths Node - Apply mask to func
        mask_func = Node(MultiImageMaths(), name = 'mask_func')
        mask_func.inputs.op_string = '-mul %s '
        run_level.connect(select_files, 'func', mask_func, 'in_file')
        run_level.connect(select_files, 'mask', mask_func, 'operand_files')

        # IsotropicSmooth Node - Smoothing data
        smoothing_func = Node(IsotropicSmooth(), name = 'smoothing_func')
        smoothing_func.inputs.fwhm = self.fwhm
        run_level.connect(mask_func, 'out_file', smoothing_func, 'in_file')

        # Get Subject Info - get subject specific condition information
        subject_information = MapNode(Function(
            function = self.get_subject_information,
            input_names = ['event_file'],
            output_names = ['subject_info'],
            iterfield=["events"]
            ), name = 'subject_information')
        run_level.connect(select_files, 'events', subject_information, 'event_file')

        # SpecifyModel Node - Generate run level model
        specify_model = MapNode(SpecifyModel(), name = 'specify_model')
        specify_model.inputs.high_pass_filter_cutoff = 100
        specify_model.inputs.input_units = 'secs'
        specify_model.inputs.time_repetition = TaskInformation()['RepetitionTime']
        run_level.connect(smoothing_func, 'out_file', specify_model, 'functional_runs')
        run_level.connect(subject_information, 'subject_info', specify_model, 'subject_info')

        # Level1Design Node - Generate files for run level computation
        model_design = MapNode(Level1Design(), name = 'model_design',  iterfield=["session_info"])
        model_design.inputs.bases = {"dgamma": {"derivs": True}}
        model_design.inputs.interscan_interval = TaskInformation()['RepetitionTime']
        model_design.inputs.model_serial_correlations = True
        model_design.inputs.contrasts = self.run_level_contrasts
        run_level.connect(specify_model, 'session_info', model_design, 'session_info')

        # FEATModel Node - Generate run level model
        model_generation = MapNode(FEATModel(), name = 'model_generation', iterfield=["fsf_file", "ev_files"])
        run_level.connect(model_design, 'ev_files', model_generation, 'ev_files')
        run_level.connect(model_design, 'fsf_files', model_generation, 'fsf_file')

        # FILMGLS Node - Estimate first level model
        model_estimate = MapNode(FILMGLS(), name='model_estimate', smooth_autocorr=True, mask_size=5, threshold=1000)
        run_level.connect(smoothing_func, 'out_file', model_estimate, 'in_file')
        run_level.connect(model_generation, 'con_file', model_estimate, 'tcon_file')
        run_level.connect(model_generation, 'design_file', model_estimate, 'design_file')
        ################
        # Nipype node
        flt = MapNode(
            interface=FLIRT(
                cost="corratio",
                dof=12,
                searchr_x=[-90, 90],
                searchr_y=[-90, 90],
                searchr_z=[-90, 90],
                interp="trilinear",
            ),
            name="example_func2standard",
            iterfield=["in_file"],
        )

        flt.inputs.reference = Info.standard_image("MNI152_T1_2mm_brain.nii.gz")
        ################
        # Connecting the node in the workflow
        run_level.connect(extract_ref, "roi_file", flt, "in_file")

        def warp_files(copes, varcopes, masks, mat):
            from nipype.interfaces import fsl

            # need to reimport here, otherwise errors come out

            out_copes = []
            out_varcopes = []
            out_masks = []

            # register mask, same function, different parameters
            warp_mask = fsl.FLIRT(apply_xfm=True, interp="nearestneighbour")
            warp_mask.inputs.reference = Info.standard_image("MNI152_T1_2mm_brain.nii.gz")
            warp_mask.inputs.in_matrix_file = mat
            warp_mask.inputs.output_type = "NIFTI_GZ"
            warp_mask.inputs.in_file = masks
            res_mask = warp_mask.run()
            out_masks.append(str(res_mask.outputs.out_file))

            # register copes & varcopes using same function, different parameters
            warp = fsl.FLIRT(apply_xfm=True, interp="trilinear")
            warp.inputs.reference = fsl.Info.standard_image("MNI152_T1_2mm_brain.nii.gz")
            warp.inputs.in_matrix_file = mat
            warp.inputs.output_type = "NIFTI_GZ"

            # register copes
            for cope in copes:
                warp.inputs.in_file = cope
                res = warp.run()
                out_copes.append(str(res.outputs.out_file))

            # register varcopes
            for varcope in varcopes:
                warp.inputs.in_file = varcope
                res = warp.run()
                out_varcopes.append(str(res.outputs.out_file))

            return out_copes, out_varcopes, out_masks


            ################
            # Nipype node
        warpfunc = MapNode(Function(
                    input_names=["copes", "varcopes", "masks", "mat"],
                    output_names=["out_copes", "out_varcopes", "out_masks"],
                    function=warp_files,
                ),
                iterfield=["copes", "varcopes", "masks", "mat"],
                name="warpfunc",
            )

            ################
            # Connecting the node in the workflow
        run_level.connect(model_estimate, "copes", warpfunc, "copes")
        run_level.connect(model_estimate, "varcopes", warpfunc, "varcopes")
        run_level.connect(dilatemask, "out_file", warpfunc, "masks")
        run_level.connect(flt, "out_matrix_file", warpfunc, "mat")


        # DataSink Node - store the wanted results in the wanted directory
        data_sink = Node(DataSink(), name = 'data_sink')
        data_sink.inputs.base_directory = self.directories.output_dir
        run_level.connect(information_source, "subject_id", data_sink, "container")
        run_level.connect(model_estimate, 'results_dir', data_sink, 'run_level_analysis.@results')
        run_level.connect(
            model_generation, 'design_file', data_sink, 'run_level_analysis.@design_file')
        run_level.connect(
            model_generation, 'design_image', data_sink, 'run_level_analysis.@design_img')
        run_level.connect([(warpfunc, data_sink, [("out_copes", "reg_copes")])])
        run_level.connect([(warpfunc, data_sink, [("out_varcopes", "reg_varcopes")])])
        run_level.connect([(warpfunc, data_sink, [("out_masks", "reg_masks")])])

        # Remove large files, if requested
        if Configuration()['pipelines']['remove_unused_data']:
            remove_masked = Node(
                InterfaceFactory.create('remove_parent_directory'),
                name = 'remove_masked')
            run_level.connect(smoothing_func, 'out_file', remove_masked, '_')
            run_level.connect(mask_func, 'out_file', remove_masked, 'file_name')

            remove_smooth = Node(
                InterfaceFactory.create('remove_parent_directory'),
                name = 'remove_smooth')
            run_level.connect(data_sink, 'out_file', remove_smooth, '_')
            run_level.connect(smoothing_func, 'out_file', remove_smooth, 'file_name')

        return run_level

    def get_run_level_outputs(self):
        """ Return the names of the files the run level analysis is supposed to generate. """

        parameters = {
            'run_id' : self.run_list,
            'subject_id' : self.subject_list,
            'contrast_id' : self.contrast_list,
        }
        parameter_sets = product(*parameters.values())
        output_dir = join(self.directories.output_dir,
            'run_level_analysis', '_run_id_{run_id}_subject_id_{subject_id}')
        templates = [
                join(output_dir, 'results', 'cope{contrast_id}.nii.gz'),
                join(output_dir, 'results', 'tstat{contrast_id}.nii.gz'),
                join(output_dir, 'results', 'varcope{contrast_id}.nii.gz'),
                join(output_dir, 'results', 'zstat{contrast_id}.nii.gz')
            ]
        return [template.format(**dict(zip(parameters.keys(), parameter_values)))\
            for parameter_values in parameter_sets for template in templates]

    def get_subject_level_analysis(self):
        """
        Create the subject level analysis workflow.

        Returns:
        - subject_level_analysis : nipype.WorkFlow
        """
        # Second level (single-subject, mean of all four scans) analysis workflow.
        subject_level = Workflow(
            base_dir = self.directories.working_dir,
            name = 'subject_level_analysis')

        # Infosource Node - To iterate on subject and runs
        information_source = Node(IdentityInterface(
            fields = ['subject_id', 'contrast_id']),
            name = 'information_source')
        information_source.iterables = [
            ('subject_id', self.subject_list),
            ('contrast_id', self.contrast_list)
            ]

        # SelectFiles node - to select necessary files
        templates = {
            "reg_copes": "*/reg_copes/*/*/cope{contr_id}_flirt.nii.gz",
            "reg_varcopes": "*/reg_varcopes/*/*/varcope{contr_id}_flirt.nii.gz",
            "reg_masks": "*/reg_masks/*/*/*.nii.gz",
        }
        select_files = Node(SelectFiles(templates), name = 'select_files')
        select_files.inputs.base_directory= self.directories.dataset_dir
        subject_level.connect(information_source, 'subject_id', select_files, 'subject_id')
        subject_level.connect(information_source, 'contrast_id', select_files, 'contrast_id')

        # Merge Node - Merge copes files for each subject
        merge_copes = Node(Merge(), name = 'merge_copes', iterfield=["in_files"])
        merge_copes.inputs.dimension = 't'
        subject_level.connect(select_files, 'cope', merge_copes, 'in_files')

        # Merge Node - Merge varcopes files for each subject
        merge_varcopes = Node(Merge(), name = 'merge_varcopes', iterfield=["in_files"])
        merge_varcopes.inputs.dimension = 't'
        subject_level.connect(select_files, 'varcope', merge_varcopes, 'in_files')

        # Split Node - Split mask list to serve them as inputs of the MultiImageMaths node.
        merge_mask = Node(
            interface=Merge(dimension="t"), iterfield=["in_files"], name="merge_mask"
        )
        ################
        # Connecting the node in the workflow
        subject_level.connect(select_files, "reg_copes", merge_copes, "in_files")
        subject_level.connect(select_files, "reg_varcopes", merge_varcopes, "in_files")
        subject_level.connect(select_files, "reg_masks", merge_mask, "in_files")

        minmask = Node(
        interface=ImageMaths(op_string="-Tmin"), iterfield=["in_file"], name="minmask"
        )

        ################
        # Connecting the node in the workflow
        subject_level.connect(merge_mask, "merged_file", minmask, "in_file")

        maskcope = Node(
            interface=ImageMaths(op_string="-mas"),
            iterfield=["in_file", "in_file2"],
            name="maskcope",
        )
        maskvarcope = Node(
            interface=ImageMaths(op_string="-mas"),
            iterfield=["in_file", "in_file2"],
            name="maskvarcope",
        )
        subject_level.connect(merge_copes, "merged_file", maskcope, "in_file")
        subject_level.connect(minmask, "out_file", maskcope, "in_file2")
        subject_level.connect(merge_varcopes, "merged_file", maskvarcope, "in_file")
        subject_level.connect(minmask, "out_file", maskvarcope, "in_file2")

        def get_contrasts_l2(in_files):
            import numpy as np

            total = len(in_files)
            # print(in_files)
            print(f"total {total}")
            n_sub = 104
            ev_list = ["ev" + str(x) for x in range(1, n_sub + 1)]
            weight_mtx = np.zeros((104, 104))
            weight_mtx = weight_mtx.astype(np.float64)
            np.fill_diagonal(weight_mtx, 1.0)
            contr = ["", "T", ev_list, list(weight_mtx[0])]
            contr = np.array(contr, dtype=object)
            contr_lst = np.tile(contr, (n_sub, 1))
            contr_lst = [list(x) for x in contr_lst]
            for i in range(n_sub):
                contr_lst[i][3] = list(weight_mtx[i])
            # print(f"contr_lst {contr_lst}")

            reg_dict = {k: None for k in ev_list}
            for k in reg_dict.keys():
                start_lst = [0.0] * total
                idx = ev_list.index(k)
                start_idx = idx * 2
                end_idx = idx * 2 + 1
                start_lst[start_idx] = 1.0
                start_lst[end_idx] = 1.0
                reg_dict[k] = start_lst
            # print(f"reg_dict {reg_dict}")

            return contr_lst, reg_dict

        ################
        # Nipype node
        contrastgen_l2 = Node(Function(
                input_names=["in_files"],
                output_names=["contr_lst", "reg_dict"],
                function=get_contrasts_l2,
            ),
            iterfield=["in_files"],
            name="contrastgen_l2",
        )

        ################
        # Connecting the node in the workflow
        subject_level.connect(select_files, "reg_copes", contrastgen_l2, "in_files")

        ################
        # Nipype node
        level2model = Node(interface=MultipleRegressDesign(), name="l2model")

        ################
        # Connecting the node in the workflow
        subject_level.connect(
            [
                (
                    contrastgen_l2,
                    level2model,
                    [("contr_lst", "contrasts"), ("reg_dict", "regressors")],
                )
            ]
        )


        # FLAMEO Node - Estimate model
        level2estimate = Node(
            interface=FLAMEO(run_mode="fe"),
            name="level2estimate",
            iterfield=["cope_file", "var_cope_file"],
        )

        ################
        # Connecting the node in the workflow
        subject_level.connect(
            [
                (maskcope, level2estimate, [("out_file", "cope_file")]),
                (maskvarcope, level2estimate, [("out_file", "var_cope_file")]),
                (minmask, level2estimate, [("out_file", "mask_file")]),
                (
                    level2model,
                    level2estimate,
                    [
                        ("design_mat", "design_file"),
                        ("design_con", "t_con_file"),
                        ("design_grp", "cov_split_file"),
                    ],
                ),
            ]
        )
        # DataSink Node - store the wanted results in the wanted directory
        datasink = Node(DataSink(), name="sinker")
        datasink.inputs.base_directory = self.directories.output_dir

        int2string = lambda x: "contrast_" + str(x)
        subject_level.connect(information_source, ("contr_id", int2string), datasink, "container")
        subject_level.connect([(level2estimate, datasink, [("stats_dir", "stats_dir")])])
        return subject_level
