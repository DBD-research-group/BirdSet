# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""BirdSet: The General Avian Monitoring Evaluation Benchmark"""

import os
import datasets
import pandas as pd 


#############################################
_BIRDSET_CITATION = """\
@article{rauch2024,
    title = {BirdSet: A Multi-Task Benchmark For Avian Diversity Monitoring},
    author={Rauch, Lukas and Schwinger, Raphael and Wirth, Moritz and Lange, Jonas and Heinrich, René},
    year={2024}
}
"""
_BIRDSET_DESCRIPTION = """\
This dataset offers a unified, well-structured platform for avian bioacoustics and consists of various tasks. \
By creating a set of tasks, BirdSet enables an overall performance score for models and uncovers their limitations \
in certain areas.
Note that each BirdSet dataset has its own citation. Please see the source to get the correct citation for each 
contained dataset. 
"""
#############################################
_SAPSUCKER_WOODS_CITATION = """\
@dataset{stefan_kahl_2022_7079380,
  author       = {Stefan Kahl and
                  Russell Charif and
                  Holger Klinck},
  title        = {{A collection of fully-annotated soundscape 
                   recordings from the Northeastern United States}},
  month        = aug,
  year         = 2022,
  publisher    = {Zenodo},
  version      = 2,
  doi          = {10.5281/zenodo.7079380},
  url          = {https://doi.org/10.5281/zenodo.7079380}
}
"""
_SAPSUCKER_WOODS_DESCRIPTION = """\
Sapsucker Woods (Kahl et al) is a ...
"""

#############################################

_AMAZON_BASIN_CITATION = """\
@dataset{w_alexander_hopping_2022_7079124,
  author       = {W. Alexander Hopping and
                  Stefan Kahl and
                  Holger Klinck},
  title        = {{A collection of fully-annotated soundscape 
                   recordings from the Southwestern Amazon Basin}},
  month        = sep,
  year         = 2022,
  publisher    = {Zenodo},
  version      = 1,
  doi          = {10.5281/zenodo.7079124},
  url          = {https://doi.org/10.5281/zenodo.7079124}
}
"""

_AMAZON_BASIN_DESCRIPTION = """\
Sapsucker Woods (Kahl et al) is a ...
"""


_BIRD_NAMES_SAPSUCKER = ['sonspa', 'scatan', 'earpoo1', 'prawar', 'wesant1', 'amerob',
       'atbtan1', 'westan', 'barswa', 'btnwar', 'gilwoo', 'rucspa1',
       'gowwar', 'amered', 'gbmgem1', 'eursta', 'attwoo1', 'oaktit',
       'woothr', 'rucspa', 'blugrb1', 'mouchi', 'motowl', 'grawar',
       'shbdow', 'hutvir', 'casspa', 'haiwoo', 'cogdov', 'purfin',
       'bhnthr1', 'norcar', 'bkhgro', 'cubthr', 'melbla1', 'blkpho',
       'brncre', 'hooori', 'killde', 'easpho', 'fatwar', 'spotow',
       'blubun', 'bbnthr1', 'borowl', 'daejun', 'chcswi1', 'scptyr1',
       'rubpep1', 'amegfi', 'amecro', 'ltwpar1', 'slbtin1', 'rewbla',
       'orcwar', 'pyrrhu', 'brebla', 'comgra', 'bkcvir1', 'houwre',
       'bkrfin', 'warvir', 'norfli', 'whiibi', 'ovenbi1', 'snoplo5',
       'lobdow', 'grycat', 'tuftit', 'lesnig', 'higtin1', 'glwgul',
       'obnthr1', 'sinqua1', 'cocwoo1', 'winwre3', 'banswa', 'wlswar',
       'crcwar', 'norwat', 'yucvir', 'yelwar1', 'gocman1', 'grekis',
       'eleeup1', 'yehpar', 'saypho', 'lesgre1', 'rebwoo', 'wilsni1',
       'chispa', 'wilfly', 'bkcchi', 'verdin', 'cantow', 'rotbec',
       'indbun', 'socfly1', 'merlin', 'moudov', 'sursco', 'houspa',
       'varbun', 'runwre3', 'mouwar', 'relpar', 'mexchi', 'herthr',
       'fiespa', 'bktspa', 'easmea', 'rudtur', 'comrav', 'woewar1',
       'hoared', 'bicthr', 'flctan', 'redcro', 'wessan', 'blumoc',
       'baleag', 'whtdov', 'rebsap', 'blujay', 'balori', 'caltow',
       'bulori', 'btywar', 'blhsis1', 'oceant1', 'sumtan', 'aldfly',
       'ruckin', 'marwre', 'brespa', 'treswa', 'houfin', 'ariwoo',
       'sincro1', 'fepowl', 'semsan', 'chclon', 'whcspa', 'swathr',
       'norhar2', 'dusgro', 'nutfly', 'hoowar', 'mallar3', 'pacwre1',
       'rtatan1', 'bkmtou1', 'y00475', 'rebnut', 'squcuc1', 'carchi',
       'oliwar', 'sora', 'wooduc', 'whwtan1', 'snobun', 'foxspa',
       'hamfly', 'rethaw', 'casfin', 'veery', 'bushti', 'cedwax',
       'barowl28', 'golvir1', 'whcsee1', 'whwcro', 'clanut', 'baywre1',
       'whtrob1', 'moutro1', 'whevir', 'eastow', 'swtkit', 'gbwwre1',
       'pinjay', 'blhsal1', 'slcsol1', 'coohaw', 'bawwar', 'bucmot2',
       'semplo', 'gockin', 'sedwre1', 'magwar', 'belkin1', 'casvir',
       'paired', 'grasal2', 'bewwre', 'gofwoo', 'cacwre', 'amgplo',
       'pregrs1', 'robgro', 'batfal1', 'buhvir', 'audwar', 'whhwre1',
       'clcrob', 'grnjay', 'naswar', 'cubcro1', 'grbwoo1', 'amebit',
       'allhum', 'doccor', 'herwar', 'nutman', 'gchwar', 'eletro',
       'bkbmag1', 'sumwre1', 'shbpig', 'horgre', 'noremt1', 'tropar',
       'stejay', 'rocpta1', 'milmac', 'wilplo', 'swaspa', 'reevir1',
       'gadwal', 'clcspa', 'cerwar', 'grnvie1', 'logshr', 'rucwar',
       'slafin1', 'astfly', 'pinsis', 'bnhcow', 'rubrob', 'grtgra',
       'easkin', 'elfowl', 'carwre', 'whbnut', 'ridrai1', 'rengre',
       'brnthr', 'gsbfin1', 'blarob1', 'buggna', 'bltwre1', 'belvir',
       'loeowl', 'yebela1', 'lobher', 'buwwar', 'plsvir', 'brcvir1',
       'normoc', 'leafly', 'purgal2', 'leasan', 'comred', 'orcori',
       'evegro', 'willet1', 'bargol', 'arcter', 'whcsee2', 'babwar',
       'blkoys', 'bkbwar', 'coukin', 'graspa', 'fiscro', 'rtlhum',
       'subfly', 'annhum', 'lesgol', 'yeejun', 'ocbfly1', 'lecspa',
       'linspa', 'olbeup1', 'layalb', 'buthum', 'gnttow', 'antnig',
       'bcptyr1', 'yebcha', 'bahmoc', 'stfgle1', 'cangoo', 'botspa',
       'lobthr', 'sltred', 'vesspa', 'souwpw1', 'bcnher', 'rutjac1',
       'thswar5', 'savspa', 'gocwoo1', 'whtswi', 'amepip', 'wbwwre1',
       'sedwre', 'stbwre1', 'acowoo', 'trsowl', 'orcpar', 'pipplo',
       'incdov', 'bubgro1', 'larspa', 'grnher', 'rcgspa1', 'comyel',
       'audori', 'foxsp2', 'plawre1', 'whwdov', 'cupcro1', 'grhowl',
       'sobtyr1', 'wtmjay1', 'brthum', 'grcfly', 'pinwar', 'scoori',
       'baispa', 'scbtan1', 'yetwar', 'truswa', 'greroa', 'gubter1',
       'olsfly', 'cibflo1', 'chwwid', 'thbvir']

#############################################


# one dataset has multiple configurations (see GLUE)
# creates different configurations for the user to select from
# builder config is inherited, has: name, version, data_dir, data_files
class BirdSetConfig(datasets.BuilderConfig):
    def __init__(
            self,
            citation,
            features=None,
            **kwargs
        ):
        super().__init__(version=datasets.Version("0.0.1"), **kwargs)

        # outsourced to reduce redundancy (otherwise has to be defined in every config)
        # note: we have to do for every task because the class label config changes!
        if features is None: 
            features = datasets.Features({
                    "audio": datasets.Audio(sampling_rate=32_000, mono=True, decode=True),
                    "label": datasets.ClassLabel(names=_BIRD_NAMES_SAPSUCKER),
                    "file": datasets.Value("string"),
                    "source": datasets.Value("string"),
                    "start_time": datasets.Value("string"), # can be changed to timestamp later
                    "end_time": datasets.Value("string"),
                    "local_time": datasets.Value("string"),
                    "events": datasets.Sequence(datasets.Value("string"))
                })
    
        self.features = features
        self.citation = citation


class BirdSet(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    BUILDER_CONFIGS = [
        BirdSetConfig(
            name="sapsucker_woods",
            description=_SAPSUCKER_WOODS_DESCRIPTION,
            citation=_SAPSUCKER_WOODS_CITATION,
            data_dir="data/NA_subset500",
            ),

        BirdSetConfig(
            name="amazon_basin",
            description=_AMAZON_BASIN_DESCRIPTION,
            citation=_AMAZON_BASIN_CITATION,
            data_dir="data/NA_amazon",
            ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description = _BIRDSET_DESCRIPTION + self.config.description,
            features = self.config.features,
            citation=self.config.citation + "\n" + _BIRDSET_CITATION,
        )

    def _split_generators(self, dl_manager):

        # download directory of the files
        dl_dir = dl_manager.download_and_extract({
                "files": os.path.join(self.config.data_dir,"files.zip"),
                "metadata": os.path.join(self.config.data_dir, "metadata.zip")
            })
        
        # overwrite features with txt (does not work, has to be in init)
        #self.config.features["label"] = datasets.ClassLabel(names_file=os.path.join(dl_dir["metadata"], "ebird_codes.txt"))
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": os.path.join(dl_dir["files"]),
                    "metapath": os.path.join(dl_dir["metadata"], "train.csv"),
                    "split": datasets.Split.TRAIN,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_dir": os.path.join(dl_dir["files"]),
                    "metapath": os.path.join(dl_dir["metadata"], "test.csv"),
                    "split": datasets.Split.TEST
                },
            ),
        ]

    def _generate_examples(self, **kwargs):
        # can maybe changed to with csv reader
        metadata = pd.read_csv(kwargs.get("metapath"))
        for key, row in metadata.iterrows():
            audio_path = os.path.join(kwargs.get("data_dir"), row["file_name"])
            yield key, {
                "audio": audio_path,
                "label": row["ebird_code"],
                "file": audio_path,
                "source": "xeno-canto",
                "start_time": None,
                "end_time": None,
                "local_time": None,
                "events": None
            }



