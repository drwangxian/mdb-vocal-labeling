"""
re-train tonet by shaun

yu labeling

adc04: 0.7789
mirex05: 0.7777
mdb (training/validation/test): oa - 0.8789/0.7353/0.7059, loss - 0.5741/1.2802/1.3337
mir1k: 0.6310
rwc: 0.7146

m2-signer labeling
adc04: 0.7516
mirex05: 0.7901
mdb: oa - 0.8704/0.7841/0.7797,  loss - 0.9206/1.2066/1.2442
mir1k: 0.6417
rwc: 0.7220
"""


DEBUG = True
GPU_ID = 0

import logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(levelname)s] %(message)s'
)
from argparse import Namespace
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
import tensorflow as tf
import torch
from model.tonet_shaun_simple import TONet
from self_defined import pytorch_set_shape_fn as _set_shape_fn
from tf_cfp import CFP as CFPClass
import numpy as np
import librosa
from self_defined import is_vocals_singer_fn, is_vocals_m2m3_fn
import medleydb as mdb
import mir_eval
from self_defined import ArrayToTableTFFn
import soundfile


if DEBUG:
    for name in logging.root.manager.loggerDict:
        if name.startswith('numba'):
            logger = logging.getLogger(name)
            logger.setLevel(logging.WARNING)

        if name.startswith('matplotlib'):
            logger = logging.getLogger(name)
            logger.setLevel(logging.WARNING)


def ssf(torch_tensor, shape, force=False):

    if DEBUG or force:
        _set_shape_fn(torch_tensor, shape)


class Config:

    def __init__(self):

        self.debug_mode = DEBUG
        Config.set_gpu_fn()

        self.allowed_labeling_methods = ('m2', 'm2-singer', 'm2m3')
        self.labeling_method = self.allowed_labeling_methods[1]

        self.train_or_inference = Namespace(
            inference='/media/ssd/music_trans/2210/4_tonet_singer_and_m2m3_labelings/singer/ckpts/d0-8',
            from_ckpt=None,
            ckpt_prefix=None
        )
        self.tb_dir = 'tb_d0_inf'

        if self.train_or_inference.inference is None:
            self.model_names = ('training', 'validation')
        else:
            self.model_names = ('training', 'validation', 'test')

        # check if tb_dir and checkpoints with the same prefix already exist
        if not self.debug_mode:
            self.chk_if_tb_dir_and_model_with_same_prefix_exist_fn()

        self.snippet_len = 128
        self.initial_learning_rate = 1e-4
        self.batch_size = 6
        self.patience_epochs = 20

        self.tvt_split_dict = Config.get_dataset_split_fn()
        if self.debug_mode:
            self.tvt_split_dict['training'] = self.tvt_split_dict['training'][:2]
            self.tvt_split_dict['validation'] = self.tvt_split_dict['validation'][:2]
            self.tvt_split_dict['test'] = self.tvt_split_dict['test'][:2]

        self.acoustic_model_ins = AcousticModel()

        if self.train_or_inference.inference is None:
            self.lr_var = torch.tensor(self.initial_learning_rate, requires_grad=False)

    @staticmethod
    def set_gpu_fn():

        gpus = tf.config.list_physical_devices('GPU')
        assert len(gpus) == 1
        tf.config.experimental.set_memory_growth(gpus[0], True)

    def chk_if_tb_dir_and_model_with_same_prefix_exist_fn(self):

        # check if tb_dir exists
        assert self.tb_dir is not None
        is_tb_dir_exist = glob.glob('{}/'.format(self.tb_dir))
        if is_tb_dir_exist:
            assert False, 'directory {} already exists'.format(self.tb_dir)

        # check if model exists
        if self.train_or_inference.inference is None and self.train_or_inference.ckpt_prefix is not None:
            ckpt_dir, ckpt_prefix = os.path.split(self.train_or_inference.ckpt_prefix)
            assert ckpt_prefix != ''
            if ckpt_dir == '':
                ckpt_dir = 'ckpts'

            is_exist = glob.glob('{}/{}*'.format(ckpt_dir, ckpt_prefix))
            if is_exist:
                assert False, 'checkpoints with prefix {} already exist'.format(ckpt_prefix)

    @staticmethod
    def get_dataset_split_fn():

        train_songlist = ["AimeeNorwich_Child", "AlexanderRoss_GoodbyeBolero", "AlexanderRoss_VelvetCurtain",
                          "AvaLuna_Waterduct", "BigTroubles_Phantom", "DreamersOfTheGhetto_HeavyLove",
                          "FacesOnFilm_WaitingForGa", "FamilyBand_Again", "Handel_TornamiAVagheggiar",
                          "HeladoNegro_MitadDelMundo", "HopAlong_SisterCities", "LizNelson_Coldwar",
                          "LizNelson_ImComingHome", "LizNelson_Rainfall", "Meaxic_TakeAStep", "Meaxic_YouListen",
                          "MusicDelta_80sRock", "MusicDelta_Beatles", "MusicDelta_Britpop", "MusicDelta_Country1",
                          "MusicDelta_Country2", "MusicDelta_Disco", "MusicDelta_Grunge", "MusicDelta_Hendrix",
                          "MusicDelta_Punk", "MusicDelta_Reggae", "MusicDelta_Rock", "MusicDelta_Rockabilly",
                          "PurlingHiss_Lolita", "StevenClark_Bounty", "SweetLights_YouLetMeDown",
                          "TheDistricts_Vermont", "TheScarletBrand_LesFleursDuMal", "TheSoSoGlos_Emergency",
                          "Wolf_DieBekherte"]
        val_songlist = ["BrandonWebster_DontHearAThing", "BrandonWebster_YesSirICanFly",
                        "ClaraBerryAndWooldog_AirTraffic", "ClaraBerryAndWooldog_Boys", "ClaraBerryAndWooldog_Stella",
                        "ClaraBerryAndWooldog_TheBadGuys", "ClaraBerryAndWooldog_WaltzForMyVictims",
                        "HezekiahJones_BorrowedHeart", "InvisibleFamiliars_DisturbingWildlife", "Mozart_DiesBildnis",
                        "NightPanther_Fire", "SecretMountains_HighHorse", "Snowmine_Curfews"]
        test_songlist = ["AClassicEducation_NightOwl", "Auctioneer_OurFutureFaces", "CelestialShore_DieForUs",
                         "Creepoid_OldTree", "Debussy_LenfantProdigue", "MatthewEntwistle_DontYouEver",
                         "MatthewEntwistle_Lontano", "Mozart_BesterJungling", "MusicDelta_Gospel",
                         "PortStWillow_StayEven", "Schubert_Erstarrung", "StrandOfOaks_Spacestation"]

        assert len(train_songlist) == 35
        assert len(val_songlist) == 13
        assert len(test_songlist) == 12

        return dict(
            training=train_songlist,
            validation=val_songlist,
            test=test_songlist
        )

    @staticmethod
    def gen_central_notes_fn():

        fmin = 32
        fmax = 2050
        bins_per_oct = 60
        central_freqs = []

        fac = 2. ** (1. / bins_per_oct)
        f = float(fmin)
        while f < fmax:
            central_freqs.append(f)
            f = f * fac
        assert len(central_freqs) == 361
        central_freqs = central_freqs[1:]
        central_freqs = librosa.hz_to_midi(central_freqs)
        assert len(central_freqs) == 360
        central_freqs = central_freqs.astype(np.float32)
        central_freqs.flags['WRITEABLE'] = False

        return central_freqs

    @staticmethod
    def get_adc04_track_ids_fn():

        test_songlist = ["daisy1", "daisy2", "daisy3", "daisy4", "opera_fem2", "opera_fem4", "opera_male3",
                         "opera_male5", "pop1", "pop2", "pop3", "pop4"]
        assert len(test_songlist) == 12

        return test_songlist

    @staticmethod
    def get_mirex05_track_ids_fn():

        test_songlist = ["train01", "train02", "train03", "train04", "train05", "train06", "train07", "train08",
                         "train09"]

        assert len(test_songlist) == 9

        return test_songlist

    @staticmethod
    def get_mir1k_track_ids_fn():

        wav_files = os.path.join(os.environ['mir1k'], 'Wavfile', '*.wav')
        wav_files = glob.glob(wav_files)
        track_ids = [os.path.basename(wav_file)[:-4] for wav_file in wav_files]
        track_ids = set(track_ids)
        assert len(track_ids) == 1000
        track_ids = list(track_ids)
        track_ids = np.sort(track_ids)
        track_ids = list(track_ids)

        return track_ids

    @staticmethod
    def get_rwc_track_ids_fn():

        track_ids = []
        for rec_idx in range(100):
            track_ids.append(str(rec_idx))

        return track_ids


class AcousticModel:

    def __init__(self):

        acoustic_model = TONet()
        self.acoustic_model = acoustic_model
        self._loss_fn_dict = dict(
            pitch=torch.nn.CrossEntropyLoss(reduction='none'),
            chroma=torch.nn.CrossEntropyLoss(reduction='none'),
            octave=torch.nn.CrossEntropyLoss(reduction='none')
        )

    def train(self):
        self.acoustic_model.train()

    def eval(self):
        self.acoustic_model.eval()

    @staticmethod
    def _cfp_to_tcfp_torch_fn(cfp):

        # cfp [None, 3, 360, 128]

        b = cfp.shape[0]
        outputs = torch.reshape(cfp, [b, 3, 6, 60, 128])
        outputs = torch.transpose(outputs, 2, 3)
        outputs = torch.reshape(outputs, [b, 3, 360, 128])

        return outputs

    def __call__(self, cfp_tf):

        cfp_tf.set_shape([None, 3, 360, 128])
        cfp = cfp_tf.numpy()
        cfp = torch.tensor(cfp, device='cuda')
        tcfp = AcousticModel._cfp_to_tcfp_torch_fn(cfp)

        logits_torch = self.acoustic_model(cfp, tcfp)

        return logits_torch

    def loss_fn(self, *, ref_notes_tf, logits_torch):

        ref_notes_tf.set_shape([None, 128])

        ssf(logits_torch['pitch'], [None, 361, 128])

        labels = AcousticModel.ref_notes_to_various_ref_labels_tf_fn(ref_notes_tf)

        loss = []
        for name in logits_torch:
            _labels = torch.tensor(labels[name].numpy(), dtype=torch.int64, device='cuda')
            _loss = self._loss_fn_dict[name](logits_torch[name], _labels)
            loss.append(_loss)
        loss = torch.stack(loss, dim=-1)
        loss = torch.mean(loss)

        return loss

    @staticmethod
    @tf.function(
        input_signature=[tf.TensorSpec([None, 128], name='ref_notes')],
        autograph=False
    )
    def ref_notes_to_various_ref_labels_tf_fn(ref_notes):

        ref_shape = [None, 128]
        ref_notes = tf.convert_to_tensor(ref_notes, dtype=tf.float32)
        ref_notes.set_shape(ref_shape)

        note_range = TFDataset.note_range
        assert len(note_range) == 360
        assert note_range[0] > 0.
        note_min, note_max = note_range[[0, -1]]

        cond_positive_pitches = ref_notes > 0.
        cond_positive_pitches.set_shape(ref_shape)
        cond = cond_positive_pitches & (ref_notes < note_min)
        ref_notes = tf.where(cond, note_min, ref_notes)

        ref_notes = tf.where(ref_notes > note_max, note_max, ref_notes)

        note_range = np.insert(note_range, 0, 0.)
        note_range = tf.convert_to_tensor(note_range, tf.float32)
        note_range.set_shape([361])

        ref_pitches = note_range[None, None, :] - ref_notes[:, :, None]
        ref_pitches.set_shape(ref_shape + [361])
        ref_pitches = ref_pitches >= 0.
        ref_pitches = tf.argmax(ref_pitches, axis=-1, output_type=tf.int32)
        ref_pitches.set_shape(ref_shape)

        B = 60
        B = tf.convert_to_tensor(B)
        ref_octaves = (ref_pitches - 1) // B + 1
        ref_octaves = tf.where(cond_positive_pitches, ref_octaves, 0)
        assert ref_octaves.dtype == tf.int32
        ref_octaves.set_shape([None, 128])

        ref_chromas = (ref_pitches - 1) % B // 5 + 1
        ref_chromas = tf.where(cond_positive_pitches, ref_chromas, 0)
        ref_chromas.set_shape(ref_shape)

        return dict(chroma=ref_chromas, pitch=ref_pitches, octave=ref_octaves)


class TFDataset:

    cfp_fn = CFPClass()

    note_range = Config.gen_central_notes_fn()
    assert len(note_range) == 360 and note_range[0] > 0.
    note_min = note_range[0]
    freq_min = librosa.midi_to_hz(note_min)

    def __init__(self, model):

        self.model = model

    @staticmethod
    def gen_spec_fn(track_id):

        wav_file = os.path.join(os.environ['medleydb'], track_id, track_id + '_MIX.wav')
        spec = TFDataset.cfp_fn(wav_file)
        ssf(spec, [3, 360, None])

        return spec

    @staticmethod
    def gen_label_yu_fn(track_id):

        yu_label_path = os.path.join(os.environ['fatnet_spec'], 'f0ref', track_id + '_MIX.txt')
        time_freqs = np.genfromtxt(yu_label_path)
        assert np.all(np.logical_not(np.isnan(time_freqs)))
        ssf(time_freqs, [None, 2])
        num_frames = len(time_freqs)
        assert time_freqs[0, 0] == 0.
        t_last = time_freqs[-1, 0]
        t_last = int(np.round(t_last / 0.01))
        assert t_last == num_frames - 1

        freqs = time_freqs[:, 1]
        TFDataset.validity_check_of_ref_freqs_fn(freqs)

        notes = TFDataset.hz_to_midi_fn(freqs)
        times = np.arange(num_frames) * 0.01

        result = dict(notes=notes, original=dict(times=times, freqs=freqs))

        return result

    @staticmethod
    def gen_label_su_or_shaun_fn(labeling_method, track_id):

        assert labeling_method in ('m2-singer', 'm2m3')
        if labeling_method == 'm2-singer':
            is_vocals_fn = is_vocals_singer_fn
        elif labeling_method == 'm2m3':
            is_vocals_fn = is_vocals_m2m3_fn
        else:
            assert False

        track = mdb.MultiTrack(track_id)
        assert not track.is_instrumental
        assert track.has_melody

        m2_melody = track.melody2_annotation
        m2_melody = np.asarray(m2_melody)
        ssf(m2_melody, [None, 2], force=True)

        is_vocals = is_vocals_fn(track_id)
        assert len(is_vocals) == len(m2_melody)

        freqs_256 = m2_melody[:, 1]
        TFDataset.validity_check_of_ref_freqs_fn(freqs_256)

        freqs_256 = np.where(is_vocals, freqs_256, 0.)
        num_frames = len(freqs_256)
        times_256 = np.arange(num_frames) * (256. / 44100)

        num_frames_441 = ((num_frames - 1) * 256 + 440) // 441 + 1
        times_441 = np.arange(num_frames_441) * 0.01
        assert times_441[-1] >= times_256[-1]
        assert times_441[-1] < times_256[-1] + 0.01
        voicing_256 = freqs_256 > TFDataset.freq_min
        freqs_441, voicing_441 = mir_eval.melody.resample_melody_series(
            times=times_256,
            frequencies=freqs_256,
            voicing=voicing_256,
            times_new=times_441
        )
        TFDataset.validity_check_of_ref_freqs_fn(freqs_441)

        notes = TFDataset.hz_to_midi_fn(freqs_441)

        result = dict(notes=notes, original=dict(times=times_256, freqs=freqs_256))

        return result

    def gen_label_fn(self, track_id):

        model = self.model
        labeling_method = model.config.labeling_method
        allowed_methods = model.config.allowed_labeling_methods
        idx = allowed_methods.index(labeling_method)

        if idx == 0:
            label = TFDataset.gen_label_yu_fn(track_id)
        else:
            label = TFDataset.gen_label_su_or_shaun_fn(labeling_method, track_id)

        return label

    @staticmethod
    def validity_check_of_ref_freqs_fn(freqs):

        min_melody_freq = TFDataset.freq_min

        all_zeros = freqs == 0.
        all_positives = freqs > min_melody_freq
        all_valid = np.logical_or(all_zeros, all_positives)
        assert np.all(all_valid)

    @staticmethod
    def hz_to_midi_fn(freqs):

        assert np.all(freqs >= 0)
        notes = np.zeros_like(freqs)
        positives = np.nonzero(freqs)
        notes[positives] = librosa.hz_to_midi(freqs[positives])

        return notes

    def gen_np_dataset_fn(self):

        model = self.model

        logging.info(f'{model.name} - generating spectrograms and labels')

        track_ids = model.config.tvt_split_dict[model.name]
        num_tracks = len(track_ids)
        dataset = []
        for track_idx, track_id in enumerate(track_ids):
            logging.debug('{}/{} - {}'.format(track_idx + 1, num_tracks, track_id))

            spec = TFDataset.gen_spec_fn(track_id)
            notes_original_dict = self.gen_label_fn(track_id)
            notes = notes_original_dict['notes']

            diff = len(notes) - spec.shape[-1]
            assert 0 <= diff <= 2
            if diff > 0:
                spec = np.pad(spec, [[0, 0], [0, 0], [0, diff]])

            spec = np.require(spec, np.float32, ['O', 'C'])
            notes = notes.astype(np.float32)
            spec.flags['WRITEABLE'] = False
            notes.flags['WRITEABLE'] = False
            dataset.append(dict(spectrogram=spec, notes=notes, original=notes_original_dict['original']))

        return dataset

    @staticmethod
    def note_out_of_range_chk_fn(np_dataset):

        note_range = TFDataset.note_range
        assert len(note_range) == 360 and note_range[0] > 0

        all_notes = [note for rec_dict in np_dataset for note in rec_dict['notes'] if note > 0]
        note_min = min(all_notes)
        note_max = max(all_notes)

        lower_note = note_range[0]
        upper_note = note_range[-1]
        logging.info('note range - ({}, {})'.format(note_min, note_max))
        if note_min < lower_note or note_min > upper_note:
            logging.warning('note min - {} - out of range'.format(note_min))
        if note_max < lower_note or note_max > upper_note:
            logging.warning('note max - {} - out of range'.format(note_max))

    @staticmethod
    def gen_split_list_fn(num_frames, snippet_len):

        split_frames = range(0, num_frames + 1, snippet_len)
        split_frames = list(split_frames)
        if split_frames[-1] != num_frames:
            split_frames.append(num_frames)
        start_end_frame_pairs = zip(split_frames[:-1], split_frames[1:])
        start_end_frame_pairs = [list(it) for it in start_end_frame_pairs]

        return start_end_frame_pairs

    def gen_rec_start_end_idx_fn(self, np_dataset):

        snippet_len = self.model.config.snippet_len
        rec_start_and_end_list = []
        for rec_idx, rec_dict in enumerate(np_dataset):
            split_list = TFDataset.gen_split_list_fn(
                num_frames=rec_dict['spectrogram'].shape[-1],
                snippet_len=snippet_len
            )
            tmp = [[rec_idx] + se for se in split_list]
            rec_start_and_end_list = rec_start_and_end_list + tmp

        return rec_start_and_end_list

    @staticmethod
    def midi_to_hz_fn(notes):

        assert np.all(notes >= 0)

        freqs = np.zeros_like(notes)
        positives = np.where(notes > 0)
        freqs[positives] = librosa.midi_to_hz(notes[positives])

        return freqs


class TFDatasetForTrainingModeTrainingSplit(TFDataset):

    def __init__(self, model):

        super(TFDatasetForTrainingModeTrainingSplit, self).__init__(model)
        assert self.model is model

        is_inferencing = model.config.train_or_inference.inference is not None
        assert not is_inferencing
        assert model.name == 'training'

        self.np_dataset = self.gen_np_dataset_fn()
        TFDataset.note_out_of_range_chk_fn(self.np_dataset)
        self.rec_start_end_idx_list = self.gen_rec_start_end_idx_fn(self.np_dataset)

        batches_per_epoch = len(self.rec_start_end_idx_list)
        batch_size = model.config.batch_size
        batches_per_epoch = (batches_per_epoch + batch_size - 1) // batch_size
        self.batches_per_epoch = batches_per_epoch

        self.tf_dataset = self.gen_tf_dataset_fn()
        self.iterator = iter(self.tf_dataset)

    def map_idx_to_data_fn(self, idx):

        n_time_frames = self.model.config.snippet_len
        np_dataset = self.np_dataset

        def py_fn(idx):

            idx = idx.numpy()
            rec_idx, start_frame, end_frame = self.rec_start_end_idx_list[idx]
            rec_dict = np_dataset[rec_idx]
            spec = rec_dict['spectrogram'][..., start_frame:end_frame]
            notes = rec_dict['notes'][start_frame:end_frame]

            t = spec.shape[-1]
            if t < n_time_frames:
                paddings = n_time_frames - t
                spec = np.pad(spec, [[0, 0], [0, 0], [0, paddings]])
                notes = np.pad(notes, [[0, paddings]])

            return spec, notes

        spec, notes = tf.py_function(py_fn, inp=[idx], Tout=[tf.float32, tf.float32])
        spec.set_shape([3, 360, n_time_frames])
        notes.set_shape([n_time_frames])

        return dict(spectrogram=spec, notes=notes)

    def gen_tf_dataset_fn(self):

        num_snippets = len(self.rec_start_end_idx_list)
        dataset = tf.data.Dataset.range(num_snippets, output_type=tf.int32)
        dataset = dataset.shuffle(num_snippets, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(self.map_idx_to_data_fn)
        batch_size = self.model.config.batch_size
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(10)

        return dataset


class TFDatasetForInferenceMode(TFDataset):

    def __init__(self, model):

        super(TFDatasetForInferenceMode, self).__init__(model)
        assert self.model is model

        is_inferencing = model.config.train_or_inference.inference is not None

        if not is_inferencing:
            assert model.name == 'validation'

        self.np_dataset = self.gen_np_dataset_fn()
        track_ids = model.config.tvt_split_dict[model.name]
        self.rec_names = tuple(track_ids)
        num_frames = [rec_dict['spectrogram'].shape[-1] for rec_dict in self.np_dataset]
        num_frames_vector = np.asarray(num_frames, dtype=np.int64)
        num_frames_vector.flags['WRITEABLE'] = False
        self.num_frames_vector = num_frames_vector
        TFDataset.note_out_of_range_chk_fn(self.np_dataset)
        self.rec_start_end_idx_list = self.gen_rec_start_end_idx_fn(self.np_dataset)
        self.batches_per_epoch = len(self.rec_start_end_idx_list)
        self.tf_dataset = self.gen_tf_dataset_fn()

    def gen_rec_start_end_idx_fn(self, np_dataset):

        model = self.model

        snippet_len = model.config.snippet_len
        batch_size = model.config.batch_size
        batch_snippet_len = batch_size * snippet_len

        rec_start_end_idx_list = []
        for rec_idx, rec_dict in enumerate(np_dataset):
            split_list = TFDataset.gen_split_list_fn(
                num_frames=rec_dict['spectrogram'].shape[-1],
                snippet_len=batch_snippet_len
            )
            rec_dict['split_list'] = split_list
            t = [[rec_idx] + se for se in split_list]
            rec_start_end_idx_list = rec_start_end_idx_list + t

        return rec_start_end_idx_list

    def map_idx_to_data_fn(self, idx):

        model = self.model

        np_dataset = self.np_dataset
        snippet_len = model.config.snippet_len
        batch_size = model.config.batch_size
        batch_snippet_len = batch_size * snippet_len

        def py_fn(idx):

            idx = idx.numpy()
            rec_idx, start_frame, end_frame = self.rec_start_end_idx_list[idx]
            rec_dict = np_dataset[rec_idx]
            assert start_frame % batch_snippet_len == 0
            batch_idx = start_frame // batch_snippet_len
            spec = rec_dict['spectrogram'][..., start_frame:end_frame]
            notes = rec_dict['notes'][start_frame:end_frame]

            effective_num_frames = end_frame - start_frame
            assert effective_num_frames > 0

            if effective_num_frames < batch_snippet_len:
                t = effective_num_frames % snippet_len
                if t > 0:
                    assert t < snippet_len
                    padded_frames = snippet_len - t
                else:
                    padded_frames = 0

                if padded_frames:
                    spec = np.pad(spec, [[0, 0], [0, 0], [0, padded_frames]])
                    notes = np.pad(notes, [[0, padded_frames]])
            elif effective_num_frames == batch_snippet_len:
                padded_frames = 0
            else:
                assert False

            assert spec.shape[-1] == len(notes)
            assert len(notes) % snippet_len == 0
            spec = np.reshape(spec, [3, 360, -1, snippet_len])
            spec = np.transpose(spec, [2, 0, 1, 3])
            notes = np.reshape(notes, [-1, snippet_len])
            assert len(spec) == len(notes)

            return rec_idx, batch_idx, padded_frames, spec, notes

        rec_idx, batch_idx, padded_frames, spec, notes = tf.py_function(
            py_fn,
            inp=[idx],
            Tout=[tf.int32, tf.int32, tf.int32, tf.float32, tf.float32]
        )

        rec_idx.set_shape([])
        batch_idx.set_shape([])
        padded_frames.set_shape([])
        spec.set_shape([None, 3, 360, snippet_len])
        notes.set_shape([None, snippet_len])

        return dict(
            rec_idx=rec_idx,
            batch_idx=batch_idx,
            padded_frames=padded_frames,
            spectrogram=spec,
            notes=notes
        )

    def gen_tf_dataset_fn(self):

        num_snippets = len(self.rec_start_end_idx_list)
        dataset = tf.data.Dataset.range(num_snippets, output_type=tf.int32)
        dataset = dataset.map(self.map_idx_to_data_fn)
        dataset = dataset.prefetch(10)

        return dataset


class TFDatasetForAdc04:

    def __init__(self, model):

        self.model = model

        is_inferencing = model.config.train_or_inference.inference is not None

        assert is_inferencing

        self.np_dataset = self.gen_np_dataset_fn()
        track_ids = model.config.tvt_split_dict[model.name]
        self.rec_names = tuple(track_ids)
        num_frames = [rec_dict['spectrogram'].shape[-1] for rec_dict in self.np_dataset]
        num_frames = np.asarray(num_frames, dtype=np.int64)
        num_frames.flags['WRITEABLE'] = False
        self.num_frames_vector = num_frames
        TFDataset.note_out_of_range_chk_fn(self.np_dataset)
        self.rec_start_end_idx_list = self.gen_rec_start_end_idx_fn(self.np_dataset)
        self.batches_per_epoch = len(self.rec_start_end_idx_list)
        self.tf_dataset = self.gen_tf_dataset_fn()

    def gen_rec_start_end_idx_fn(self, np_dataset):

        model = self.model

        snippet_len = model.config.snippet_len
        batch_size = model.config.batch_size
        batch_snippet_len = batch_size * snippet_len

        rec_start_end_idx_list = []
        for rec_idx, rec_dict in enumerate(np_dataset):
            split_list = TFDataset.gen_split_list_fn(
                num_frames=rec_dict['spectrogram'].shape[-1],
                snippet_len=batch_snippet_len
            )
            rec_dict['split_list'] = split_list
            t = [[rec_idx] + se for se in split_list]
            rec_start_end_idx_list = rec_start_end_idx_list + t

        return rec_start_end_idx_list

    def gen_spec_fn(self, track_id):

        mix_wav_file = os.path.join(os.environ['adc04'], track_id + '.wav')

        return TFDataset.cfp_fn(mix_wav_file)

    def gen_label_fn(self, track_id):
        # the reference melody uses a hop size of 256/44100 samples
        melody2_suffix = 'REF.txt'

        annot_path = os.path.join(os.environ['adc04'], track_id + melody2_suffix)
        times_labels = np.genfromtxt(annot_path, delimiter=None)
        assert np.all(np.logical_not(np.isnan(times_labels)))
        ssf(times_labels, [None, 2], force=True)
        num_frames = len(times_labels)
        t = times_labels[-1, 0]
        t = int(round(t / (256. / 44100.)))
        assert t + 1 == num_frames
        assert times_labels[0, 0] == 0.

        freqs = times_labels[:, 1]
        TFDataset.validity_check_of_ref_freqs_fn(freqs)

        times_256 = np.arange(num_frames) * (256. / 44100.)
        num_frames_10ms = ((num_frames - 1) * 256 + 440) // 441 + 1
        times_10ms = np.arange(num_frames_10ms) * 0.010
        assert times_10ms[-1] >= times_256[-1]
        assert times_10ms[-1] < times_256[-1] + 0.01
        freqs_10ms, _ = mir_eval.melody.resample_melody_series(
            times=times_256,
            frequencies=freqs,
            voicing=freqs > .1,
            times_new=times_10ms
        )
        TFDataset.validity_check_of_ref_freqs_fn(freqs_10ms)
        notes = TFDataset.hz_to_midi_fn(freqs_10ms)

        result = dict(notes=notes, original=dict(times=times_labels[:, 0], freqs=freqs))

        return result

    def gen_np_dataset_fn(self):

        assert not hasattr(self, 'np_dataset')

        model = self.model

        logging.info('{} - adc04 - generate spectrograms and labels'.format(model.name))

        config = model.config
        track_ids = config.tvt_split_dict[model.name]
        num_tracks = len(track_ids)
        dataset = []
        for track_idx, track_id in enumerate(track_ids):
            logging.debug('{}/{}'.format(track_idx + 1, num_tracks))

            spec = self.gen_spec_fn(track_id)
            ssf(spec, [3, 360, None], force=True)
            notes_original_dict = self.gen_label_fn(track_id)
            notes = notes_original_dict['notes']

            diff = len(notes) - spec.shape[-1]
            assert 0 <= diff <= 1
            if diff == 1:
                spec = np.pad(spec, [[0, 0], [0, 0], [0, 1]])

            spec = np.require(spec, np.float32, ['O', 'C'])
            notes = notes.astype(np.float32)
            spec.flags['WRITEABLE'] = False
            notes.flags['WRITEABLE'] = False
            dataset.append(dict(spectrogram=spec, notes=notes, original=notes_original_dict['original']))

        return dataset

    def map_idx_to_data_fn(self, idx):

        model = self.model

        np_dataset = self.np_dataset
        snippet_len = model.config.snippet_len
        batch_size = model.config.batch_size
        batch_snippet_len = batch_size * snippet_len

        def py_fn(idx):

            idx = idx.numpy()
            rec_idx, start_frame, end_frame = self.rec_start_end_idx_list[idx]
            rec_dict = np_dataset[rec_idx]
            assert start_frame % batch_snippet_len == 0
            batch_idx = start_frame // batch_snippet_len
            spec = rec_dict['spectrogram'][..., start_frame:end_frame]
            notes = rec_dict['notes'][start_frame:end_frame]

            effective_num_frames = end_frame - start_frame
            assert effective_num_frames > 0

            if effective_num_frames < batch_snippet_len:
                t = effective_num_frames % snippet_len
                if t > 0:
                    assert t < snippet_len
                    padded_frames = snippet_len - t
                else:
                    padded_frames = 0

                if padded_frames:
                    spec = np.pad(spec, [[0, 0], [0, 0], [0, padded_frames]])
                    notes = np.pad(notes, [[0, padded_frames]])
            elif effective_num_frames == batch_snippet_len:
                padded_frames = 0
            else:
                assert False

            assert spec.shape[-1] == len(notes)
            assert len(notes) % snippet_len == 0
            spec = np.reshape(spec, [3, 360, -1, snippet_len])
            spec = np.transpose(spec, [2, 0, 1, 3])
            notes = np.reshape(notes, [-1, snippet_len])
            assert len(spec) == len(notes)

            return rec_idx, batch_idx, padded_frames, spec, notes

        rec_idx, batch_idx, padded_frames, spec, notes = tf.py_function(
            py_fn,
            inp=[idx],
            Tout=[tf.int32, tf.int32, tf.int32, tf.float32, tf.float32]
        )

        rec_idx.set_shape([])
        batch_idx.set_shape([])
        padded_frames.set_shape([])
        spec.set_shape([None, 3, 360, snippet_len])
        notes.set_shape([None, snippet_len])

        return dict(
            rec_idx=rec_idx,
            batch_idx=batch_idx,
            padded_frames=padded_frames,
            spectrogram=spec,
            notes=notes
        )

    def gen_tf_dataset_fn(self):

        num_snippets = len(self.rec_start_end_idx_list)
        dataset = tf.data.Dataset.range(num_snippets, output_type=tf.int32)
        dataset = dataset.map(self.map_idx_to_data_fn)
        dataset = dataset.prefetch(10)

        return dataset


class TFDatasetForMirex05(TFDatasetForAdc04):

    def __init__(self, model):

        super(TFDatasetForMirex05, self).__init__(model)

    def gen_spec_fn(self, track_id):

        mix_wav_file = os.path.join(os.environ['mirex05'], track_id + '.wav')

        return TFDataset.cfp_fn(mix_wav_file)

    def gen_label_fn(self, track_id):
        # reference melody uses a hop size of 441 samples, or 10 ms

        if track_id == 'train13MIDI':
            m2_file = os.path.join(os.environ['mirex05'], 'train13REF.txt')
        else:
            m2_file = os.path.join(os.environ['mirex05'], track_id + 'REF.txt')
        times_labels = np.genfromtxt(m2_file, delimiter=None)
        assert np.all(np.logical_not(np.isnan(times_labels)))
        ssf(times_labels, [None, 2], force=True)
        num_frames = len(times_labels)
        t = times_labels[-1, 0]
        t = int(round(t / .01))
        assert t + 1 == num_frames
        assert times_labels[0, 0] == 0.
        freqs_441 = times_labels[:, 1]
        TFDataset.validity_check_of_ref_freqs_fn(freqs_441)
        notes = TFDataset.hz_to_midi_fn(freqs_441)
        result = dict(notes=notes, original=dict(times=times_labels[:, 0], freqs=freqs_441))

        return result

    def gen_np_dataset_fn(self):

        assert not hasattr(self, 'np_dataset')

        model = self.model

        logging.info('{} - Mirex05 training - generate spectrograms and labels'.format(model.name))

        config = model.config
        track_ids = config.tvt_split_dict[model.name]
        num_tracks = len(track_ids)
        dataset = []
        for track_idx, track_id in enumerate(track_ids):
            logging.info('{}/{} - {}'.format(track_idx + 1, num_tracks, track_id))

            spec = self.gen_spec_fn(track_id)
            ssf(spec, [3, 360, None])
            notes_original_dict = self.gen_label_fn(track_id)
            notes = notes_original_dict['notes']

            diff = len(notes) - spec.shape[-1]
            logging.info('  diff - {}'.format(diff))
            if diff > 0:
                spec = np.pad(spec, [[0, 0], [0, 0], [0, diff]])
            elif diff < 0:
                notes = np.pad(notes, [[0, -diff]])

            assert len(notes) == spec.shape[-1]

            spec = np.require(spec, np.float32, ['O', 'C'])
            notes = notes.astype(np.float32)
            spec.flags['WRITEABLE'] = False
            notes.flags['WRITEABLE'] = False
            dataset.append(dict(spectrogram=spec, notes=notes, original=notes_original_dict['original']))

        return dataset


class TFDatasetForMir1k(TFDatasetForAdc04):

    def __init__(self, model):

        super(TFDatasetForMir1k, self).__init__(model)

    def gen_spec_fn(self, track_id):

        wav_file = os.path.join(os.environ['mir1k'], 'Wavfile', track_id + '.wav')
        spec = TFDataset.cfp_fn(wav_file)

        return spec

    def gen_label_fn(self, track_id):
        # reference melody uses a hop size 20 ms. the staring time is 20 ms instead of 0 ms.

        wav_file = os.path.join(os.environ['mir1k'], 'Wavfile', track_id + '.wav')
        wav_info = soundfile.info(wav_file)
        assert wav_info.samplerate == 16000
        num_samples = wav_info.frames

        pitch_file = os.path.join(os.environ['mir1k'], 'PitchLabel', track_id + '.pv')
        pitches = np.genfromtxt(pitch_file)
        assert np.all(np.logical_not(np.isnan(pitches)))
        assert pitches.ndim == 1
        num_frames = len(pitches)
        w = 640
        assert num_samples >= w
        _num_frames = (num_samples - w) // 320 + 1
        assert num_frames == _num_frames
        t1 = pitches > TFDataset.note_min
        t2 = pitches == 0
        assert np.all(np.logical_or(t1, t2))

        num_frames = num_frames + 1
        times_20ms = np.arange(num_frames) * .02
        pitches = np.pad(pitches, [[1, 0]])
        assert len(pitches) == len(times_20ms) == num_frames

        num_frames_10ms = (num_frames - 1) * 2 + 1
        times_10ms = np.arange(num_frames_10ms) * 0.010
        assert times_10ms[-1] == times_20ms[-1]
        pitches_10ms, _ = mir_eval.melody.resample_melody_series(
            times=times_20ms,
            frequencies=pitches,
            voicing=pitches > .1,
            times_new=times_10ms
        )
        t1 = pitches_10ms == 0
        t2 = pitches_10ms > TFDataset.note_min
        assert np.all(np.logical_or(t1, t2))
        freqs_20ms = TFDataset.midi_to_hz_fn(pitches)

        result = dict(notes=pitches_10ms, original=dict(times=times_20ms, freqs=freqs_20ms))

        return result

    def gen_np_dataset_fn(self):

        assert not hasattr(self, 'np_dataset')

        model = self.model

        logging.info('{} - mir1k - generate spectrograms and labels'.format(model.name))

        config = model.config
        track_ids = config.tvt_split_dict[model.name]
        num_tracks = len(track_ids)
        dataset = []
        diffs = []
        for track_idx, track_id in enumerate(track_ids):
            logging.info('{}/{} - {}'.format(track_idx + 1, num_tracks, track_id))

            spec = self.gen_spec_fn(track_id)
            ssf(spec, [3, 360, None], force=True)
            notes_original_dict = self.gen_label_fn(track_id)
            notes = notes_original_dict['notes']

            diff = len(notes) - spec.shape[-1]
            if diff not in diffs:
                diffs.append(diff)
            if diff > 0:
                spec = np.pad(spec, [[0, 0], [0, 0], [0, diff]])
            elif diff < 0:
                notes = np.pad(notes, [[0, -diff]])

            assert len(notes) == spec.shape[-1]

            spec = np.require(spec, np.float32, ['O', 'C'])
            notes = notes.astype(np.float32)
            spec.flags['WRITEABLE'] = False
            notes.flags['WRITEABLE'] = False
            dataset.append(dict(spectrogram=spec, notes=notes, original=notes_original_dict['original']))

        diffs = np.asarray(diffs)
        max_diff = diffs.max()
        min_diff = diffs.min()
        logging.info('diff - max - {} - min - {}'.format(max_diff, min_diff))

        return dataset


class TFDatasetForRWC:

    def __init__(self, model):

        self.model = model

        is_inferencing = model.config.train_or_inference.inference is not None
        assert is_inferencing

        self.rec_files = TFDatasetForRWC.get_rec_files_fn()
        self.np_dataset = self.gen_np_dataset_fn()
        track_ids = model.config.tvt_split_dict[model.name]
        self.rec_names = tuple(track_ids)
        num_frames = [rec_dict['spectrogram'].shape[-1] for rec_dict in self.np_dataset]
        num_frames = np.asarray(num_frames, dtype=np.int64)
        num_frames.flags['WRITEABLE'] = False
        self.num_frames_vector = num_frames
        TFDataset.note_out_of_range_chk_fn(self.np_dataset)
        self.rec_start_end_idx_list = self.gen_rec_start_end_idx_fn(self.np_dataset)
        self.batches_per_epoch = len(self.rec_start_end_idx_list)
        self.tf_dataset = self.gen_tf_dataset_fn()

    @staticmethod
    def get_rec_files_fn():

        dir_prefix = 'RWC-MDB-P-2001-M0'
        dir_path = os.path.join(os.environ['rwc'], 'popular', dir_prefix)
        num_recordings = []
        for dir_idx in range(7):
            dir_idx = dir_idx + 1
            disk_dir = dir_path + str(dir_idx)
            aiff_files = glob.glob(os.path.join(disk_dir, '*.aiff'))
            num_recordings.append(len(aiff_files))
        start_end_list = np.cumsum(num_recordings)
        assert start_end_list[-1] == 100
        start_end_list = np.pad(start_end_list, [[1, 0]])

        rec_files = []
        for rec_idx in range(100):
            disk_idx = np.searchsorted(start_end_list, rec_idx, side='right')
            assert disk_idx >= 1
            disk_idx = disk_idx - 1
            disk_path = dir_path + str(disk_idx + 1)
            disk_start_rec_idx = start_end_list[disk_idx]
            rec_idx_within_disk = rec_idx - disk_start_rec_idx
            rec_idx_within_disk = rec_idx_within_disk + 1

            recs = glob.glob(os.path.join(disk_path, '*.aiff'))
            assert len(recs) == num_recordings[disk_idx]
            for recording_name in recs:
                t = os.path.basename(recording_name)
                t = t.split()[0]
                if t == str(rec_idx_within_disk):
                    rec_files.append(recording_name)
                    break
            else:
                assert False
        assert len(rec_files) == 100
        t = set(rec_files)
        assert len(t) == 100

        return rec_files

    def get_num_frames_fn(self, track_id):

        assert isinstance(track_id, str)
        rec_idx = int(track_id)
        assert rec_idx >= 0
        aiff_file = self.rec_files[rec_idx]
        aiff_info = soundfile.info(aiff_file)
        assert aiff_info.samplerate == 44100
        num_samples = aiff_info.frames
        h = 441
        num_frames = (num_samples + h - 1) // h

        return num_frames

    def gen_spec_fn(self, track_id):

        assert isinstance(track_id, str)
        rec_idx = int(track_id)
        assert rec_idx >= 0
        aiff_file = self.rec_files[rec_idx]

        return TFDataset.cfp_fn(aiff_file)

    def load_melody_from_file_fn(self, track_id):

        assert isinstance(track_id, str)
        rec_idx = int(track_id)
        melody_dir = os.path.join(os.environ['rwc'], 'popular', 'AIST.RWC-MDB-P-2001.MELODY')
        melody_prefix = 'RM-P'
        melody_suffix = '.MELODY.TXT'
        melody_file = melody_prefix + str(rec_idx + 1).zfill(3) + melody_suffix
        melody_file = os.path.join(melody_dir, melody_file)

        with open(melody_file, 'r') as fh:
            lines = fh.readlines()
            line = lines[-1]
            cols = line.split()
            num_frames = int(cols[0]) + 1
            aiff_num_frames = self.get_num_frames_fn(track_id)
            assert num_frames <= aiff_num_frames
            freqs = np.zeros([aiff_num_frames], np.float32)
            min_freq = 31.
            for line in lines:
                cols = line.split()
                assert len(cols) == 5
                assert cols[0] == cols[1]
                assert cols[2] == 'm'
                frame_idx = int(cols[0])
                assert frame_idx >= 0
                freq = float(cols[3])
                assert freq == 0 or freq > min_freq
                freqs[frame_idx] = freq

            return freqs

    def gen_label_fn(self, track_id):

        freqs_441 = self.load_melody_from_file_fn(track_id)
        num_frames_441 = len(freqs_441)
        times_441 = np.arange(num_frames_441) * 0.01
        TFDataset.validity_check_of_ref_freqs_fn(freqs_441)
        notes = TFDataset.hz_to_midi_fn(freqs_441)

        result = dict(notes=notes, original=dict(times=times_441, freqs=freqs_441))

        return result

    def gen_np_dataset_fn(self):

        assert not hasattr(self, 'np_dataset')

        model = self.model

        logging.info('{} - rwc - generate spectrograms and labels'.format(model.name))

        config = model.config
        track_ids = config.tvt_split_dict[model.name]
        num_tracks = len(track_ids)
        dataset = []
        for track_idx, track_id in enumerate(track_ids):
            logging.info('{}/{} - {}'.format(track_idx + 1, num_tracks, track_id))

            spec = self.gen_spec_fn(track_id)
            ssf(spec, [3, 360, None], force=True)
            notes_original_dict = self.gen_label_fn(track_id)
            notes = notes_original_dict['notes']

            diff = len(notes) - spec.shape[-1]
            assert np.abs(diff) <= 1
            if diff == 1:
                spec = np.pad(spec, [[0, 0], [0, 0], [0, 1]])
            elif diff == -1:
                notes = np.pad(notes, [[0, 1]])
            assert len(notes) == spec.shape[-1]

            spec = np.require(spec, np.float32, ['O', 'C'])
            notes = notes.astype(np.float32)
            spec.flags['WRITEABLE'] = False
            notes.flags['WRITEABLE'] = False
            dataset.append(dict(spectrogram=spec, notes=notes, original=notes_original_dict['original']))

        return dataset

    def gen_rec_start_end_idx_fn(self, np_dataset):

        model = self.model

        snippet_len = model.config.snippet_len
        batch_size = model.config.batch_size
        batch_snippet_len = batch_size * snippet_len

        rec_start_end_idx_list = []
        for rec_idx, rec_dict in enumerate(np_dataset):
            split_list = TFDataset.gen_split_list_fn(
                num_frames=rec_dict['spectrogram'].shape[-1],
                snippet_len=batch_snippet_len
            )
            rec_dict['split_list'] = split_list
            t = [[rec_idx] + se for se in split_list]
            rec_start_end_idx_list = rec_start_end_idx_list + t

        return rec_start_end_idx_list

    def map_idx_to_data_fn(self, idx):

        model = self.model

        np_dataset = self.np_dataset
        snippet_len = model.config.snippet_len
        batch_size = model.config.batch_size
        batch_snippet_len = batch_size * snippet_len

        def py_fn(idx):

            idx = idx.numpy()
            rec_idx, start_frame, end_frame = self.rec_start_end_idx_list[idx]
            rec_dict = np_dataset[rec_idx]
            assert start_frame % batch_snippet_len == 0
            batch_idx = start_frame // batch_snippet_len
            spec = rec_dict['spectrogram'][..., start_frame:end_frame]
            notes = rec_dict['notes'][start_frame:end_frame]

            effective_num_frames = end_frame - start_frame
            assert effective_num_frames > 0

            if effective_num_frames < batch_snippet_len:
                t = effective_num_frames % snippet_len
                if t > 0:
                    assert t < snippet_len
                    padded_frames = snippet_len - t
                else:
                    padded_frames = 0

                if padded_frames:
                    spec = np.pad(spec, [[0, 0], [0, 0], [0, padded_frames]])
                    notes = np.pad(notes, [[0, padded_frames]])
            elif effective_num_frames == batch_snippet_len:
                padded_frames = 0
            else:
                assert False

            assert spec.shape[-1] == len(notes)
            assert len(notes) % snippet_len == 0
            spec = np.reshape(spec, [3, 360, -1, snippet_len])
            spec = np.transpose(spec, [2, 0, 1, 3])
            notes = np.reshape(notes, [-1, snippet_len])
            assert len(spec) == len(notes)

            return rec_idx, batch_idx, padded_frames, spec, notes

        rec_idx, batch_idx, padded_frames, spec, notes = tf.py_function(
            py_fn,
            inp=[idx],
            Tout=[tf.int32, tf.int32, tf.int32, tf.float32, tf.float32]
        )

        rec_idx.set_shape([])
        batch_idx.set_shape([])
        padded_frames.set_shape([])
        spec.set_shape([None, 3, 360, snippet_len])
        notes.set_shape([None, snippet_len])

        return dict(
            rec_idx=rec_idx,
            batch_idx=batch_idx,
            padded_frames=padded_frames,
            spectrogram=spec,
            notes=notes
        )

    def gen_tf_dataset_fn(self):

        num_snippets = len(self.rec_start_end_idx_list)
        dataset = tf.data.Dataset.range(num_snippets, output_type=tf.int32)
        dataset = dataset.map(self.map_idx_to_data_fn)
        dataset = dataset.prefetch(10)

        return dataset


class MetricsBase:

    @staticmethod
    def count_nonzero_fn(inputs):

        outputs = inputs
        outputs = tf.math.count_nonzero(outputs, dtype=tf.int32)
        outputs = tf.cast(outputs, tf.int64)

        return outputs

    @staticmethod
    def octave(distance):

        distance = tf.floor(distance / 12. + .5) * 12.

        return distance

    @staticmethod
    def to_f8_divide_and_to_f4_fn(numerator, denominator):

        numerator = tf.cast(numerator, tf.float64)
        denominator = tf.cast(denominator, tf.float64)
        numerator = numerator / tf.maximum(denominator, 1e-7)
        numerator = tf.cast(numerator, tf.float32)

        return numerator

    @staticmethod
    @tf.function(
        input_signature=[tf.TensorSpec(None, dtype=tf.int32, name='pred_indices')]
    )
    def est_notes_from_pred_indices_tf_fn(pred_indices):

        note_range = TFDataset.note_range
        assert len(note_range) == 360
        assert note_range[0] > 0
        note_range = np.insert(note_range, 0, 0.)
        note_range = tf.convert_to_tensor(note_range, tf.float32)
        pred_notes = tf.gather(note_range, pred_indices)

        return pred_notes


class MetricsTrainingModeTrainingSplit:

    def __init__(self, model):

        assert model.config.train_or_inference.inference is None
        assert model.name == 'training'

        self.model = model
        self.var_dict = self.define_tf_variables_fn()

        self.oa = None
        self.loss = None

    def reset(self):

        for var in self.var_dict['all_updated']:
            var = var.deref()
            var.assign(tf.zeros_like(var))

        self.oa = None
        self.loss = None

    def update_melody_var_fn(self, l1, l2, value):

        assert l1 is not None
        assert value is not None

        var_dict = self.var_dict['melody']
        all_updated = self.var_dict['all_updated']
        if l2 is not None:
            var = var_dict[l1][l2]
        else:
            var = var_dict[l1]

        var_ref = var.ref()
        assert not all_updated[var_ref]

        var.assign_add(value)
        all_updated[var_ref] = True

    def update_loss_fn(self, value):

        assert value is not None

        all_updated = self.var_dict['all_updated']
        var = self.var_dict['loss']
        var_ref = var.ref()
        assert not all_updated[var_ref]
        assert value.dtype == var.dtype
        var.assign_add(value)
        all_updated[var_ref] = True

    def increase_batch_counter_fn(self):

        all_updated = self.var_dict['all_updated']
        var = self.var_dict['batch_counter']
        var_ref = var.ref()
        assert not all_updated[var_ref]
        var.assign_add(1)
        all_updated[var_ref] = True

    def define_tf_variables_fn(self):

        model = self.model

        with tf.name_scope(model.name):
            with tf.name_scope('statistics'):

                all_defined_vars_updated = dict()

                def gen_tf_var(name):

                    assert name

                    with tf.device('/cpu:0'):
                        var = tf.Variable(
                            initial_value=tf.zeros([], dtype=tf.int64),
                            trainable=False,
                            name=name
                        )
                    var_ref = var.ref()
                    assert var_ref not in all_defined_vars_updated
                    all_defined_vars_updated[var_ref] = False

                    return var

                melody_var_dict = dict(
                    gt=dict(
                        voiced=gen_tf_var('voiced'),
                        unvoiced=gen_tf_var('unvoiced')
                    ),
                    voicing=dict(
                        correct_voiced=gen_tf_var('correct_voiced'),
                        incorrect_voiced=gen_tf_var('incorrect_voiced'),
                        correct_unvoiced=gen_tf_var('correct_unvoiced')
                    ),
                    correct_pitches=gen_tf_var('correct_pitches'),
                    correct_chromas=gen_tf_var('correct_chromas')
                )

                batch_counter = tf.Variable(
                    initial_value=tf.zeros([], dtype=tf.int32),
                    trainable=False,
                    name='batch_counter'
                )
                all_defined_vars_updated[batch_counter.ref()] = False
                loss = tf.Variable(
                    initial_value=tf.zeros([], tf.float32),
                    trainable=False,
                    name='acc_loss'
                )
                all_defined_vars_updated[loss.ref()] = False

                return dict(
                    melody=melody_var_dict,
                    batch_counter=batch_counter,
                    loss=loss,
                    all_updated=all_defined_vars_updated
                )

    @tf.function(input_signature=[
        tf.TensorSpec([None, 128], name='ref_notes'),
        tf.TensorSpec([None, 128], dtype=tf.int32, name='pred_indices'),
        tf.TensorSpec([], name='loss')
    ], autograph=False)
    def _update_states_tf_fn(self, ref_notes, pred_indices, loss):

        count_nz_fn = MetricsBase.count_nonzero_fn

        ref_notes = tf.convert_to_tensor(ref_notes, tf.float32)
        ref_notes.set_shape([None, 128])

        pred_indices = tf.convert_to_tensor(pred_indices, tf.int32)
        pred_indices.set_shape([None, 128])

        ref_voicing = ref_notes > .1
        n_ref_voicing = tf.logical_not(ref_voicing)

        est_voicing = pred_indices > 0
        n_est_voicing = tf.logical_not(est_voicing)

        voiced = count_nz_fn(ref_voicing)
        self.update_melody_var_fn('gt', 'voiced', voiced)
        unvoiced = tf.size(ref_voicing, tf.int64) - voiced
        self.update_melody_var_fn('gt', 'unvoiced', unvoiced)
        cond_both_voiced = ref_voicing & est_voicing
        correct_voiced = count_nz_fn(cond_both_voiced)
        self.update_melody_var_fn('voicing', 'correct_voiced', correct_voiced)
        incorrect_voiced = count_nz_fn(n_ref_voicing & est_voicing)
        self.update_melody_var_fn('voicing', 'incorrect_voiced', incorrect_voiced)
        correct_unvoiced = count_nz_fn(n_ref_voicing & n_est_voicing)
        self.update_melody_var_fn('voicing', 'correct_unvoiced', correct_unvoiced)

        pred_notes = MetricsBase.est_notes_from_pred_indices_tf_fn(pred_indices)

        est_ref_note_diffs = tf.abs(pred_notes - ref_notes)

        correct_pitches = est_ref_note_diffs < .5
        correct_pitches = count_nz_fn(correct_pitches & cond_both_voiced)
        self.update_melody_var_fn('correct_pitches', None, correct_pitches)

        correct_chromas = est_ref_note_diffs
        octave = MetricsBase.octave(correct_chromas)
        correct_chromas = tf.abs(correct_chromas - octave) < .5
        correct_chromas = count_nz_fn(correct_chromas & cond_both_voiced)
        self.update_melody_var_fn('correct_chromas', None, correct_chromas)

        self.update_loss_fn(loss)
        self.increase_batch_counter_fn()

        assert all(self.var_dict['all_updated'].values())

    def update_states(self, ref_notes_tf, pitch_logits_torch, loss_torch):

        if DEBUG:
            assert isinstance(ref_notes_tf, tf.Tensor)
            t = [pitch_logits_torch, loss_torch]
            assert all(torch.is_tensor(v) for v in t)

            ref_notes_tf.set_shape([None, 128])
            ssf(pitch_logits_torch, [None, 361, 128])

        pred_indices = torch.argmax(pitch_logits_torch, dim=1)
        pred_indices = pred_indices.cpu().numpy().astype(np.int32)

        loss = loss_torch.item()

        self._update_states_tf_fn(
            ref_notes=ref_notes_tf,
            pred_indices=pred_indices,
            loss=loss
        )

    def results(self):

        melody_dict = self.var_dict['melody']
        var_loss = self.var_dict['loss']
        var_batch_counter = self.var_dict['batch_counter']
        f8f4div = MetricsBase.to_f8_divide_and_to_f4_fn

        num_frames = melody_dict['gt']['voiced'] + melody_dict['gt']['unvoiced']
        m_oa = f8f4div(melody_dict['correct_pitches'] + melody_dict['voicing']['correct_unvoiced'], num_frames)
        m_oa.set_shape([])

        m_vrr = f8f4div(melody_dict['voicing']['correct_voiced'], melody_dict['gt']['voiced'])
        m_vfa = f8f4div(melody_dict['voicing']['incorrect_voiced'], melody_dict['gt']['unvoiced'])
        m_va = f8f4div(
            melody_dict['voicing']['correct_voiced'] + melody_dict['voicing']['correct_unvoiced'],
            num_frames
        )
        m_rpa = f8f4div(melody_dict['correct_pitches'], melody_dict['gt']['voiced'])
        m_rca = f8f4div(melody_dict['correct_chromas'], melody_dict['gt']['voiced'])
        m_loss = var_loss / tf.cast(var_batch_counter, tf.float32)

        self.oa = m_oa.numpy()
        self.loss = m_loss.numpy()

        results = dict(
            loss=m_loss,
            vrr=m_vrr,
            vfa=m_vfa,
            va=m_va,
            rpa=m_rpa,
            rca=m_rca,
            oa=m_oa
        )

        return results


class MetricsInference:

    def __init__(self, model):

        self.model = model
        self.num_recs = len(model.config.tvt_split_dict[model.name])

        self.oa = None
        self.loss = None
        self.rec_idx = None
        self.batch_idx = None
        self.mir_eval_oas = []
        self.tf_oas = None

        is_inferencing = model.config.train_or_inference.inference is not None
        if not is_inferencing:
            assert model.name == 'validation'

        self.var_dict = self.define_tf_variables_fn()

    def define_tf_variables_fn(self):

        model = self.model
        num_recs = self.num_recs

        with tf.name_scope(model.name):
            with tf.name_scope('statistics'):

                all_defined_vars_updated = dict()

                def gen_tf_var(name):

                    assert name

                    with tf.device('/cpu:0'):
                        var = tf.Variable(
                            initial_value=tf.zeros([num_recs], dtype=tf.int64),
                            trainable=False,
                            name=name
                        )
                    var_ref = var.ref()
                    assert var_ref not in all_defined_vars_updated
                    all_defined_vars_updated[var_ref] = False

                    return var

                melody_var_dict = dict(
                    gt=dict(
                        voiced=gen_tf_var('voiced'),
                        unvoiced=gen_tf_var('unvoiced')
                    ),
                    voicing=dict(
                        correct_voiced=gen_tf_var('correct_voiced'),
                        incorrect_voiced=gen_tf_var('incorrect_voiced'),
                        correct_unvoiced=gen_tf_var('correct_unvoiced')
                    ),
                    correct_pitches=gen_tf_var('correct_pitches'),
                    correct_chromas=gen_tf_var('correct_chromas')
                )

                batch_counter = tf.Variable(
                    initial_value=tf.zeros([], dtype=tf.int32),
                    trainable=False,
                    name='batch_counter'
                )
                all_defined_vars_updated[batch_counter.ref()] = False
                loss = tf.Variable(
                    initial_value=tf.zeros([], tf.float32),
                    trainable=False,
                    name='acc_loss'
                )
                all_defined_vars_updated[loss.ref()] = False

                return dict(
                    melody=melody_var_dict,
                    batch_counter=batch_counter,
                    loss=loss,
                    all_updated=all_defined_vars_updated
                )

    def reset(self):

        for var in self.var_dict['all_updated']:
            var = var.deref()
            var.assign(tf.zeros_like(var))

        self.oa = None
        self.loss = None
        self.rec_idx = None
        self.batch_idx = None
        self.mir_eval_oas = []
        self.tf_oas = None

    def update_melody_var_fn(self, rec_idx, l1, l2, value):

        assert rec_idx is not None
        assert l1 is not None
        assert value is not None

        var_dict = self.var_dict['melody']
        all_updated = self.var_dict['all_updated']
        if l2 is not None:
            var = var_dict[l1][l2]
        else:
            var = var_dict[l1]

        var_ref = var.ref()
        assert not all_updated[var_ref]

        with tf.device(var.device):
            value = tf.identity(value)
        var.scatter_add(tf.IndexedSlices(values=value, indices=rec_idx))
        all_updated[var_ref] = True

    def update_loss_fn(self, value):

        assert value is not None

        all_updated = self.var_dict['all_updated']
        var = self.var_dict['loss']
        var_ref = var.ref()
        assert not all_updated[var_ref]
        assert value.dtype == var.dtype
        var.assign_add(value)
        all_updated[var_ref] = True

    def increase_batch_counter_fn(self):

        all_updated = self.var_dict['all_updated']
        var = self.var_dict['batch_counter']
        var_ref = var.ref()
        assert not all_updated[var_ref]
        var.assign_add(1)
        all_updated[var_ref] = True

    @tf.function(
        input_signature=[
            tf.TensorSpec([], dtype=tf.int32, name='rec_idx'),
            tf.TensorSpec([None, 128], name='ref_notes'),
            tf.TensorSpec([None, 128], dtype=tf.int32, name='pred_indices'),
            tf.TensorSpec([], name='loss'),
            tf.TensorSpec([], dtype=tf.int32, name='padded_frames')
        ], autograph=False
    )
    def _update_states_tf_fn(self, rec_idx, ref_notes, pred_indices, loss, padded_frames):

        count_nonzero_fn = MetricsBase.count_nonzero_fn

        rec_idx = tf.convert_to_tensor(rec_idx, tf.int32)
        rec_idx.set_shape([])

        ref_notes = tf.convert_to_tensor(ref_notes, tf.float32)
        ref_notes.set_shape([None, 128])

        pred_indices = tf.convert_to_tensor(pred_indices, tf.int32)
        pred_indices.set_shape([None, 128])

        loss = tf.convert_to_tensor(loss, tf.float32)
        loss.set_shape([])

        padded_frames = tf.convert_to_tensor(padded_frames, tf.int32)
        padded_frames.set_shape([])

        ref_notes = tf.reshape(ref_notes, [-1])
        pred_indices = tf.reshape(pred_indices, [-1])

        padded = padded_frames > 0
        ref_notes = tf.cond(padded, lambda: ref_notes[:-padded_frames], lambda: ref_notes)
        pred_indices = tf.cond(padded, lambda: pred_indices[:-padded_frames], lambda: pred_indices)

        ref_voicing = ref_notes > .1
        n_ref_voicing = tf.logical_not(ref_voicing)

        est_notes = MetricsBase.est_notes_from_pred_indices_tf_fn(pred_indices)
        est_voicing = pred_indices > 0
        n_est_voicing = tf.logical_not(est_voicing)

        est_ref_note_diffs = tf.abs(est_notes - ref_notes)
        est_ref_note_diffs.set_shape([None])

        voiced_frames = count_nonzero_fn(ref_voicing)
        unvoiced_frames = tf.size(ref_voicing, tf.int64) - voiced_frames
        ref_voicing_logical_and_est_voicing = ref_voicing & est_voicing
        correct_voiced_frames = count_nonzero_fn(ref_voicing_logical_and_est_voicing)
        incorrect_voiced_frames = count_nonzero_fn(n_ref_voicing & est_voicing)
        correct_unvoiced_frames = count_nonzero_fn(n_ref_voicing & n_est_voicing)
        self.update_melody_var_fn(rec_idx, 'gt', 'voiced', voiced_frames)
        self.update_melody_var_fn(rec_idx, 'gt', 'unvoiced', unvoiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'correct_voiced', correct_voiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'incorrect_voiced', incorrect_voiced_frames)
        self.update_melody_var_fn(rec_idx, 'voicing', 'correct_unvoiced', correct_unvoiced_frames)

        correct_pitches = est_ref_note_diffs < .5
        correct_pitches = count_nonzero_fn(correct_pitches & ref_voicing_logical_and_est_voicing)
        self.update_melody_var_fn(rec_idx, 'correct_pitches', None, correct_pitches)

        correct_chromas = est_ref_note_diffs
        octave = MetricsBase.octave(correct_chromas)
        correct_chromas = tf.abs(correct_chromas - octave) < .5
        correct_chromas = count_nonzero_fn(correct_chromas & ref_voicing_logical_and_est_voicing)
        self.update_melody_var_fn(rec_idx, 'correct_chromas', None, correct_chromas)

        self.update_loss_fn(loss)
        self.increase_batch_counter_fn()

        assert all(self.var_dict['all_updated'].values())

        return est_notes

    def update_states(self, rec_idx_tf, batch_idx_tf, padded_frames_tf, ref_notes_tf, pitch_logits_torch, loss_torch):

        model = self.model

        if DEBUG:
            t = [rec_idx_tf, batch_idx_tf, padded_frames_tf, ref_notes_tf]
            assert all(isinstance(v, tf.Tensor) for v in t)

            t = [pitch_logits_torch, loss_torch]
            assert all(torch.is_tensor(v) for v in t)

            ssf(pitch_logits_torch, [None, 361, 128])

        _rec_idx = rec_idx_tf.numpy()
        rec_dict = model.tf_dataset.np_dataset[_rec_idx]
        num_batches = len(rec_dict['split_list'])
        _batch_idx = batch_idx_tf.numpy()
        assert _batch_idx < num_batches

        padded_frames = padded_frames_tf.numpy()
        if _batch_idx < num_batches - 1:
            assert padded_frames == 0

        if _batch_idx == 0:
            self.rec_idx = _rec_idx
            self.batch_idx = _batch_idx
            self.est_notes = []

        assert _rec_idx == self.rec_idx

        if _batch_idx > 0:
            assert _batch_idx == self.batch_idx + 1

        self.batch_idx = _batch_idx

        pred_indices = torch.argmax(pitch_logits_torch, dim=1)
        pred_indices = pred_indices.cpu().numpy().astype(np.int32)

        est_notes = self._update_states_tf_fn(
            rec_idx=rec_idx_tf,
            ref_notes=ref_notes_tf,
            pred_indices=pred_indices,
            loss=loss_torch.item(),
            padded_frames=padded_frames_tf
        ).numpy()
        self.est_notes.append(est_notes)

        if _batch_idx == num_batches - 1:
            est_notes = np.concatenate(self.est_notes)
            num_frames = model.tf_dataset.num_frames_vector[_rec_idx]
            assert est_notes.shape == (num_frames,)
            oa = self.mir_eval_oa_fn(_rec_idx, est_notes)
            self.mir_eval_oas.append(oa)

    @staticmethod
    def est_notes_to_hz_fn(est_notes):

        min_note = TFDataset.note_min

        larger = est_notes >= min_note
        equal = est_notes == 0.
        larger_or_equal = np.logical_or(larger, equal)
        assert np.all(larger_or_equal)

        positives = np.where(est_notes >= min_note)

        freqs = np.zeros_like(est_notes)
        freqs[positives] = librosa.midi_to_hz(est_notes[positives])

        return freqs

    def mir_eval_oa_fn(self, rec_idx, est_notes):

        model = self.model

        rec_dict = model.tf_dataset.np_dataset[rec_idx]
        ref_times = rec_dict['original']['times']
        ref_freqs = rec_dict['original']['freqs']
        assert len(ref_times) == len(ref_freqs)

        est_freqs = MetricsInference.est_notes_to_hz_fn(est_notes)

        if np.all(est_freqs == 0.):
            logging.warning('{} - all frames unvoiced'.format(rec_idx))

        num_frames = len(est_freqs)
        est_times = np.arange(num_frames) * 0.01

        oa = mir_eval.melody.evaluate(
            ref_time=ref_times,
            ref_freq=ref_freqs,
            est_time=est_times,
            est_freq=est_freqs
        )['Overall Accuracy']

        return oa

    def results(self):

        model = self.model
        num_recs = self.num_recs
        melody_dict = self.var_dict['melody']
        var_loss = self.var_dict['loss']
        var_batch_counter = self.var_dict['batch_counter']
        f8f4div = MetricsBase.to_f8_divide_and_to_f4_fn
        num_frames_vector = tf.convert_to_tensor(model.tf_dataset.num_frames_vector, tf.int64)

        _num_frames_vector = melody_dict['gt']['voiced'] + melody_dict['gt']['unvoiced']
        tf.debugging.assert_equal(_num_frames_vector, num_frames_vector)

        correct_frames = melody_dict['correct_pitches'] + melody_dict['voicing']['correct_unvoiced']
        m_oa = f8f4div(correct_frames, num_frames_vector)
        m_oa.set_shape([num_recs])

        m_vrr = f8f4div(melody_dict['voicing']['correct_voiced'], melody_dict['gt']['voiced'])
        m_vfa = f8f4div(melody_dict['voicing']['incorrect_voiced'], melody_dict['gt']['unvoiced'])
        m_va = f8f4div(
            melody_dict['voicing']['correct_voiced'] + melody_dict['voicing']['correct_unvoiced'],
            num_frames_vector
        )
        m_rpa = f8f4div(
            melody_dict['correct_pitches'], melody_dict['gt']['voiced']
        )
        m_rca = f8f4div(
            melody_dict['correct_chromas'], melody_dict['gt']['voiced']
        )
        m_loss = var_loss / tf.cast(var_batch_counter, tf.float32)

        self.tf_oas = m_oa.numpy()
        self.oa = tf.reduce_mean(m_oa).numpy()
        self.loss = m_loss.numpy()

        results = dict(
            loss=m_loss,
            vrr=m_vrr,
            vfa=m_vfa,
            va=m_va,
            rpa=m_rpa,
            rca=m_rca,
            oa=m_oa
        )

        return results


class TBSummary:

    def __init__(self, model):

        assert hasattr(model, 'metrics')

        self.model = model
        self.is_inferencing = model.config.train_or_inference.inference is not None

        if not self.is_inferencing:
            assert 'test' not in model.name

        self.tb_path = os.path.join(model.config.tb_dir, model.name)
        self.tb_summary_writer = tf.summary.create_file_writer(self.tb_path)

        if hasattr(model.tf_dataset, 'rec_names'):
            self.rec_names = model.tf_dataset.rec_names
            self.num_recs = len(self.rec_names)

        self.header = ['vrr', 'vfa', 'va', 'rpa', 'rca', 'oa']
        self.num_columns = len(self.header)

        self.table_ins = self.create_tf_table_writer_ins_fn()

    def create_tf_table_writer_ins_fn(self):

        is_inferencing = self.is_inferencing
        model = self.model
        header = self.header
        description = 'metrics'
        tb_summary_writer = self.tb_summary_writer

        if hasattr(self, 'rec_names'):
            assert is_inferencing or not is_inferencing and model.name == 'validation'
            names = list(self.rec_names) + ['average']
            table_ins = ArrayToTableTFFn(
                writer=tb_summary_writer,
                header=header,
                scope=description,
                title=description,
                names=names
            )
        else:
            assert not is_inferencing and model.is_training
            table_ins = ArrayToTableTFFn(
                writer=tb_summary_writer,
                header=header,
                scope=description,
                title=description,
                names=['average']
            )

        return table_ins

    def prepare_table_data_fn(self, result_dict):

        header = self.header

        if hasattr(self, 'rec_names'):
            data = [result_dict[name] for name in header]
            data = tf.stack(data, axis=-1)
            tf.ensure_shape(data, [self.num_recs, self.num_columns])
            ave = tf.reduce_mean(data, axis=0, keepdims=True)
            data = tf.concat([data, ave], axis=0)
        else:
            data = [result_dict[name] for name in header]
            data = [data]
            data = tf.convert_to_tensor(data)

        return data

    def write_tb_summary_fn(self, step_int):

        model = self.model
        is_inferencing = self.is_inferencing

        assert isinstance(step_int, int)

        with tf.name_scope(model.name):
            with tf.name_scope('statistics'):

                result_dict = model.metrics.results()

                if not is_inferencing:
                    with self.tb_summary_writer.as_default():
                        for metric_name in ('loss', 'oa'):
                            value = getattr(model.metrics, metric_name)
                            assert value is not None
                            tf.summary.scalar(metric_name, value, step=step_int)
                else:
                    with self.tb_summary_writer.as_default():
                        loss = model.metrics.loss
                        assert loss is not None
                        tf.summary.text('loss', str(loss), step=step_int)

                data = self.prepare_table_data_fn(result_dict)
                self.table_ins.write(data, step_int)


class Model:

    def __init__(self, config, name):

        assert name in config.model_names

        inferencing = config.train_or_inference.inference is not None

        if not inferencing:
            assert 'test' not in name

        self.name = name
        self.is_training = name == 'training'
        self.config = config

        if inferencing:
            if self.name == 'test':
                self.tf_dataset = TFDatasetForInferenceMode(self)
            else:
                self.tf_dataset = TFDatasetForInferenceMode(self)
        else:
            if self.is_training:
                self.tf_dataset = TFDatasetForTrainingModeTrainingSplit(self)
            else:
                self.tf_dataset = TFDatasetForInferenceMode(self)

        if inferencing:
            self.metrics = MetricsInference(self)
        else:
            if self.is_training:
                self.metrics = MetricsTrainingModeTrainingSplit(self)
            else:
                self.metrics = MetricsInference(self)

        self.tb_summary_ins = TBSummary(self)


def main():

    MODEL_DICT = {}
    MODEL_DICT['config'] = Config()
    for name in MODEL_DICT['config'].model_names:
        MODEL_DICT[name] = Model(config=MODEL_DICT['config'], name=name)

    aug_info = []
    aug_info.append('labeling method - {}'.format(MODEL_DICT['config'].labeling_method))
    aug_info.append('tb dir - {}'.format(MODEL_DICT['config'].tb_dir))
    aug_info.append('debug mode - {}'.format(MODEL_DICT['config'].debug_mode))
    if MODEL_DICT['config'].train_or_inference.inference is None:
        aug_info.append('batch size - {}'.format(MODEL_DICT['config'].batch_size))
        t = MODEL_DICT['training'].tf_dataset.batches_per_epoch
        aug_info.append('num of batches per epoch - {}'.format(t))
        aug_info.append('num of patience epochs - {}'.format(MODEL_DICT['config'].patience_epochs))
        aug_info.append('initial learning rate - {}'.format(MODEL_DICT['config'].initial_learning_rate))
    aug_info = '\n\n'.join(aug_info)
    logging.info(aug_info)
    with MODEL_DICT['training'].tb_summary_ins.tb_summary_writer.as_default():
        tf.summary.text('auxiliary_information', aug_info, step=0)

    def training_fn(global_step=None):

        assert isinstance(global_step, int)

        config = MODEL_DICT['config']
        model = MODEL_DICT['training']

        assert config.train_or_inference.inference is None

        iterator = model.tf_dataset.iterator
        acoustic_model = config.acoustic_model_ins
        acoustic_model.train()
        metrics = model.metrics
        write_tb_summary_fn = model.tb_summary_ins.write_tb_summary_fn
        batches_per_epoch = model.tf_dataset.batches_per_epoch
        optimizer = model.config.optimizer
        loss_fn = acoustic_model.loss_fn

        metrics.reset()
        for batch_idx in range(batches_per_epoch):
            logging.debug('batch {}/{}'.format(batch_idx + 1, batches_per_epoch))
            batch = iterator.get_next()

            ref_notes_tf = batch['notes']
            spec_tf = batch['spectrogram']

            logits_dict_torch = acoustic_model(cfp_tf=spec_tf)
            loss_torch = loss_fn(
                ref_notes_tf=ref_notes_tf,
                logits_torch=logits_dict_torch
            )
            optimizer.zero_grad()
            loss_torch.backward()
            optimizer.step()
            metrics.update_states(
                ref_notes_tf=ref_notes_tf,
                pitch_logits_torch=logits_dict_torch['pitch'],
                loss_torch=loss_torch
            )
        write_tb_summary_fn(global_step)

        loss = model.metrics.loss
        oa = model.metrics.oa
        logging.info(f'{model.name} - step - {global_step} - loss - {loss} - oa - {oa}')

    def validation_and_inference_fn(model_name, global_step=None):

        config = MODEL_DICT['config']
        is_inferencing = config.train_or_inference.inference is not None

        if not is_inferencing:
            assert model_name == 'validation'

        assert isinstance(global_step, int)

        model = MODEL_DICT[model_name]

        acoustic_model = config.acoustic_model_ins
        acoustic_model.eval()
        assert not hasattr(model.tf_dataset, 'iterator')
        iterator = iter(model.tf_dataset.tf_dataset)
        metrics = model.metrics
        batches_per_epoch = model.tf_dataset.batches_per_epoch
        loss_fn = acoustic_model.loss_fn

        metrics.reset()
        with torch.no_grad():
            for batch_idx in range(batches_per_epoch):
                batch = iterator.get_next()

                ref_notes_tf = batch['notes']
                spec = batch['spectrogram']

                logits_dict_torch = acoustic_model(cfp_tf=spec)
                loss_torch = loss_fn(
                    ref_notes_tf=ref_notes_tf,
                    logits_torch=logits_dict_torch
                )
                metrics.update_states(
                    rec_idx_tf=batch['rec_idx'],
                    batch_idx_tf=batch['batch_idx'],
                    padded_frames_tf=batch['padded_frames'],
                    ref_notes_tf=ref_notes_tf,
                    pitch_logits_torch=logits_dict_torch['pitch'],
                    loss_torch=loss_torch
                )
        batch = iterator.get_next_as_optional()
        assert not batch.has_value()

        model.tb_summary_ins.write_tb_summary_fn(global_step)

        loss = model.metrics.loss
        oa = model.metrics.oa
        logging.info(f'{model.name} - step - {global_step} - loss - {loss} - oa - {oa}')

        if is_inferencing:
            mir_eval_oas = metrics.mir_eval_oas
            mir_eval_oas = np.asarray(mir_eval_oas)
            tf_oas = metrics.tf_oas

            oa_diffs = tf_oas - mir_eval_oas

            print('tf and mir_eval oas and their differences -')
            for idx in range(len(tf_oas)):
                print(idx, tf_oas[idx], mir_eval_oas[idx], oa_diffs[idx])
            tf_oa = np.mean(tf_oas)
            mir_eval_oa = np.mean(mir_eval_oas)
            diff = tf_oa - mir_eval_oa
            print('ave', tf_oa, mir_eval_oa, diff)

    def complete_ckpt_file_fn(ckpt_file):

        ckpt_dir, ckpt_name = os.path.split(ckpt_file)
        if ckpt_dir == '':
            ckpt_dir = 'ckpts'
            ckpt_file = os.path.join(ckpt_dir, ckpt_name)
        ckpt_file = ckpt_file + '.tar'

        return ckpt_file

    if MODEL_DICT['config'].train_or_inference.inference is not None:
        ckpt_file = MODEL_DICT['config'].train_or_inference.inference
        ckpt_file = complete_ckpt_file_fn(ckpt_file)
        state_dict = torch.load(ckpt_file)
        acoustic_model_ins = MODEL_DICT['config'].acoustic_model_ins
        acoustic_model = acoustic_model_ins.acoustic_model
        acoustic_model.load_state_dict(state_dict['acoustic_model'], strict=True)
        acoustic_model = acoustic_model.cuda()
        acoustic_model_ins.acoustic_model = acoustic_model

        logging.info('inferencing with checkpoint {}'.format(ckpt_file))
        for model_name in MODEL_DICT['config'].model_names:
            logging.info(model_name)
            validation_and_inference_fn(model_name, global_step=0)
            MODEL_DICT[model_name].tb_summary_ins.tb_summary_writer.close()

    elif MODEL_DICT['config'].train_or_inference.from_ckpt is not None:
        ckpt_file = MODEL_DICT['config'].train_or_inference.from_ckpt

        logging.info('resume training from {}'.format(ckpt_file))

        ckpt_file = complete_ckpt_file_fn(ckpt_file)
        state_dict = torch.load(ckpt_file)

        acoustic_model_ins = MODEL_DICT['config'].acoustic_model_ins
        acoustic_model = acoustic_model_ins.acoustic_model
        acoustic_model.load_state_dict(state_dict['acoustic_model'], strict=True)

        acoustic_model = acoustic_model.cuda()
        acoustic_model_ins.acoustic_model = acoustic_model

        assert not hasattr(MODEL_DICT['config'], 'optimizer')
        optimizer = torch.optim.Adam(acoustic_model.parameters(), lr=MODEL_DICT['config'].lr_var)
        optimizer.load_state_dict(state_dict['optimizer'])
        MODEL_DICT['config'].optimizer = optimizer

        logging.info('reproducing results ...')

        model_name = 'validation'
        logging.info(model_name)
        validation_and_inference_fn(model_name, global_step=0)
        best_oa = MODEL_DICT[model_name].metrics.oa
        best_epoch = 0

    else:
        logging.info('training from scratch ...')

        acoustic_model_ins = MODEL_DICT['config'].acoustic_model_ins
        acoustic_model = acoustic_model_ins.acoustic_model
        acoustic_model.cuda()
        acoustic_model_ins.acoustic_model = acoustic_model

        assert not hasattr(MODEL_DICT['config'], 'optimizer')
        optimizer = torch.optim.Adam(acoustic_model.parameters(), lr=MODEL_DICT['config'].lr_var)
        MODEL_DICT['config'].optimizer = optimizer

        best_oa = None

    # training
    if MODEL_DICT['config'].train_or_inference.inference is None:
        assert MODEL_DICT['config'].train_or_inference.ckpt_prefix is not None
        ckpt_dir, ckpt_prefix = os.path.split(MODEL_DICT['config'].train_or_inference.ckpt_prefix)
        assert ckpt_prefix != ''
        if ckpt_dir == '':
            ckpt_dir = 'ckpts'
        if not os.path.exists(ckpt_dir):
            os.system('mkdir {}'.format(ckpt_dir))

        patience_epochs = MODEL_DICT['config'].patience_epochs
        training_epoch = 1

        while True:

            logging.info('\nepoch - {}'.format(training_epoch))

            for model_name in MODEL_DICT['config'].model_names:

                logging.info(model_name)

                if model_name == 'training':
                    training_fn(training_epoch)
                elif model_name == 'validation':
                    validation_and_inference_fn(model_name, training_epoch)

            valid_oa = MODEL_DICT['validation'].metrics.oa
            should_save = best_oa is None or valid_oa > best_oa
            if should_save:
                best_oa = valid_oa
                best_epoch = training_epoch
                existing_ckpts = glob.glob(os.path.join(ckpt_dir, ckpt_prefix + '*.tar'))
                if len(existing_ckpts) >= 1:
                    for existing_ckpt in existing_ckpts:
                        os.system('rm {}'.format(existing_ckpt))
                acoustic_model = MODEL_DICT['config'].acoustic_model_ins.acoustic_model
                state_dict = dict(
                    acoustic_model=acoustic_model.state_dict(),
                    optimizer=MODEL_DICT['config'].optimizer.state_dict()
                )
                ckpt_file_name = os.path.join(ckpt_dir, ckpt_prefix + '-{}.tar'.format(training_epoch))
                torch.save(state_dict, ckpt_file_name)
                logging.info('weights checkpointed to {}'.format(ckpt_file_name))

            d = training_epoch - best_epoch
            if d >= patience_epochs:
                logging.info('training terminated at epoch {}'.format(training_epoch))
                break

            training_epoch = training_epoch + 1

        for model_name in MODEL_DICT['config'].model_names:
            model = MODEL_DICT[model_name]
            model.tb_summary_ins.tb_summary_writer.close()

        if 'ramdisk' in ckpt_dir and not DEBUG:
            tmp_ckpt_dir = ckpt_dir
            ckpt_dir = ckpt_dir.split('/')[-1]
            assert len(ckpt_dir) > 0
            if os.path.isdir(ckpt_dir):
                os.system(f'rm -r {ckpt_dir}')
            if tmp_ckpt_dir[-1] == '/':
                tmp_ckpt_dir = tmp_ckpt_dir[:-1]
            os.system(f'cp -r {tmp_ckpt_dir} {ckpt_dir}')
            logging.info(f'ckpt copied from {tmp_ckpt_dir} to {ckpt_dir}')
            os.system(f'rm -r {tmp_ckpt_dir}')


if __name__ == '__main__':
    main()




