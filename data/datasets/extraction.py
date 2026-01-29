import os
import json
import math
import random

import numpy as np
import librosa
import soundfile as sf

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from data.datasets.prompt_templates import build_style_description

SR = 16000

ANIMAL_LABELS = [
    'owl hooting',
    'baltimore oriole calling',
    'pheasant crowing',
    'turkey gobbling',
    'canary calling',
    'cow lowing',
    'fox barking',
    'cattle mooing',
    'cricket chirping',
    'dog growling',
    'lions roaring',
    'chicken clucking',
    'frog croaking',
    'pig oinking',
    'cat hissing',
    'elk bugling',
    'chicken crowing',
    'gibbon howling',
    'black capped chickadee calling',
    'chipmunk chirping',
    'dinosaurs bellowing',
    'chimpanzee pant-hooting',
    'cat purring',
    'cat meowing',
    'chinchilla barking',
    'whale calling',
    'duck quacking',
    'dog barking',
    'woodpecker pecking tree',
    'eagle screaming',
    'sheep bleating',
    'snake hissing',
    'dog howling',
    'mynah bird singing',
    'dog baying',
    'crow cawing',
    'magpie calling',
    'mosquito buzzing',
    'ferret dooking',
    'cheetah chirrup',
    'dog whimpering',
    'elephant trumpeting',
    'snake rattling',
    'francolin calling',
    'lions growling',
    'sea lion barking',
    'otter growling',
    'goose honking',
    'goat bleating',
    'dog bow-wow',
    'coyote howling',
    'cat growling',
    'penguins braying',
    'cat caterwauling',
    'cuckoo bird calling',
    'bird squawking',
    'zebra braying',
]

def get_animal_labels_in_vggsound():
    return ANIMAL_LABELS

def extract_prompt(label):
    if isinstance(label, tuple) or isinstance(label, list):
        label = str(label[0])
    return f'Please extract the sound of {label}.'

def remove_prompt(label):
    if isinstance(label, tuple) or isinstance(label, list):
        label = str(label[0])
    return f'Please remove the sound of {label}.'

class AudioMixtures(Dataset):

    def __init__(
        self, 
        data_root,
        manifest_files,
        select_n=0,
        keep_spks=True,
        filt_labels=None,
        filt_labels_mode='both'
    ):
        super().__init__()
        self.data_root = data_root
        self.manifest_files = manifest_files
        print(f'Fetched {len(manifest_files)} manifest files.')

        self.records = []
        for manifest_file in manifest_files:
            with open(manifest_file, 'r') as json_file:
                self.records += json.load(json_file)

        if filt_labels != None:
            print(f'Filtering test set by labels... {filt_labels_mode} audio(s) should belong to labels...')
            if filt_labels_mode == 'both':
                records = [x for x in self.records if x['label3'] in filt_labels and x['label4'] in filt_labels]
            elif filt_labels_mode == 'either':
                raise NotImplementedError
                records = [x for x in self.records if x['label3'] in filt_labels or x['label4'] in filt_labels]
            self.records = records
            print(f'Found {str(len(self.records))} mixtures from {str(len(filt_labels))} labels.')

        if select_n > 0:
            self.records = self.records[:select_n]

        self.keep_spks = keep_spks

    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):

        record = self.records[idx]
        x = {}

        n_src = 4 if 'path4' in record else \
            3 if 'path3' in record else \
            2 if 'path2' in record else \
            ValueError(-1)
        assert n_src == 4

        if self.keep_spks:
            mix_path = record['path'].replace('<DATA_ROOT>', self.data_root)
            src_paths = [record[f'path{i}'].replace('<DATA_ROOT>', self.data_root)
                for i in range(1, n_src+1)]

            # Load input mixture
            mix, sr = sf.read(mix_path, dtype='float32')
            x['mix'] = mix

            # Load every source
            srcs = [sf.read(src_path, dtype='float32')[0] for src_path in src_paths]

            x['speeches'] = srcs[0:2]
            x['audios'] = srcs[2:4]
            x['speech_labels'] = [
                build_style_description(
                    gender=record['gender1'],
                    pitch=record['pitch1'],
                    tempo=record['tempo1'],
                    energy=record['energy1'],
                    emotion=record['emotion1']
                ),
                build_style_description(
                    gender=record['gender2'],
                    pitch=record['pitch2'],
                    tempo=record['tempo2'],
                    energy=record['energy2'],
                    emotion=record['emotion2']
                )
            ]
            x['audio_labels'] = [record['label3'], record['label4']]

        else:
            src_paths = [record[f'path{i}'].replace('<DATA_ROOT>', self.data_root)
                for i in [3, 4]]

            # Load audio sources
            srcs = [sf.read(src_path, dtype='float32')[0] for src_path in src_paths]

            mix = srcs[0] + srcs[1]
            x['mix'] = mix
            x['audios'] = srcs
            x['audio_labels'] = [record['label3'], record['label4']]

        return x


class TrainAudioMixtures(AudioMixtures):
        
    def __getitem__(self, idx):
        x = super().__getitem__(idx)
        what_to_do = random.choice(
            ['TAE1', 'TAE2', 'TAR1', 'TAR2']
        )
        if what_to_do == 'TAE1':
            x['tar'] = x['audios'][0]
            x['prompt'] = extract_prompt(x['audio_labels'][0])
        elif what_to_do == 'TAE2':
            x['tar'] = x['audios'][1]
            x['prompt'] = extract_prompt(x['audio_labels'][1])
        elif what_to_do == 'TAR1':
            x['tar'] = x['mix'] - x['audios'][0]
            x['prompt'] = remove_prompt(x['audio_labels'][0])
        elif what_to_do == 'TAR2':
            x['tar'] = x['mix'] - x['audios'][1]
            x['prompt'] = remove_prompt(x['audio_labels'][1])

        x['task'] = what_to_do[:3]

        return x

class TestAudioMixtures(AudioMixtures):
        
    def __getitem__(self, idx, what_to_do=None):
        x = super().__getitem__(idx)
        if what_to_do == None:
            what_to_do = ['TAE1', 'TAE2', 'TAR1', 'TAR2'][idx%4]

        if what_to_do == 'TAE1':
            x['tar'] = x['audios'][0]
            x['prompt'] = extract_prompt(x['audio_labels'][0])
        elif what_to_do == 'TAE2':
            x['tar'] = x['audios'][1]
            x['prompt'] = extract_prompt(x['audio_labels'][1])
        elif what_to_do == 'TAR1':
            x['tar'] = x['mix'] - x['audios'][0]
            x['prompt'] = remove_prompt(x['audio_labels'][0])
        elif what_to_do == 'TAR2':
            x['tar'] = x['mix'] - x['audios'][1]
            x['prompt'] = remove_prompt(x['audio_labels'][1])

        x['task'] = what_to_do[:3]

        return x
