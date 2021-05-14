
# Korean FastSpeech 2 - Pytorch Implementation

![](./assets/model.png)
# Introduction

Fastspeech2는 기존의 자기회귀(Autoregressive) 기반의 느린 학습 및 합성 속도를 개선한 모델입니다. 비자기회귀(Non Autoregressive) 기반의 모델로, Variance Adaptor에서 분산 데이터들을 통해, speech 예측의 정확도를 높일 수 있습니다.
즉 기존의 audio-text만으로 예측을 하는 모델에서, pitch,energy,duration을 추가한 모델입니다. 
Fastspeech2에서 duration은 MFA(Montreal Forced Aligner)를 통해 추출합니다. 이렇게 추출한 duration을 바탕으로 phoneme(음소)와 음성간의 alignment가 만들어집니다.  


* 본 Repository는 https://github.com/HGU-DLLAB/Korean-FastSpeech2-Pytorch 를 변형 및 수정한 것입니다.


#### 변경 사항

- hparms.py 내 textgrid_path 변수 추가
- preprocess.py 내 Textgrid.zip의 위치를 옮기는 코드 삭제
- preprocess.py 내 "wavs"와 "wav_bak" 생성시, 발생하는  exist error 수정 (shutill.rmtree)  
- 그 외 각종 버그 수정


# Install Dependencies

먼저 python=3.7, [pytorch](https://pytorch.org/)=1.6, [ffmpeg](https://ffmpeg.org/)와 [g2pk](https://github.com/Kyubyong/g2pK)를 설치합니다.
```
# ffmpeg install
sudo apt-get install ffmpeg

# [WARNING] g2pk를 설치하시기 전에, g2pk github을 참조하셔서 g2pk의 dependency를 설치하시고 g2pk를 설치하시기 바랍니다.
pip install g2pk
```

다음으로, 필요한 모듈을 pip를 이용하여 설치합니다.
```
pip install -r requirements.txt
```

# Preprocessing

**(1) 데이터셋 구조 확인**
* 해당 프로젝트는 KSS 데이터셋을 기반으로 작성되었습니다. ([Korean-Single-Speech dataset](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)) 따라서 KSS 데이터셋과 똑같은 데이터 구조를 맞추어 주는 것이, 진행하기에 편리합니다.

![캡처](https://user-images.githubusercontent.com/63226383/117932810-cbe29a80-b33b-11eb-9093-4814f3449262.PNG)

데이터셋은 위와 같은 구조를 이루어야 합니다. 데이터셋안에 1,2,3,4 라는 sub_directory와 폴더 내의 각 wav파일과 그에 대한 transcript을 기록한 Metadata(transcript.v.1.4.txt)가 있어야 합니다.

이 중 1번 폴더는 valid set으로 사용되고, 나머지 폴더 train set으로 사용됩니다. (향후 train-test split을 이용해 변경예정)

**(2) MFA Train**

* MFA(Montreal Forced Aligner)는 Fastspeech2 학습에 반드시 필요한, Duration을 추출하기 위해 사용됩니다. MFA는 발화(음성 파일)와 Phoneme sequence간의 alignment를 실행하고 이를 TextGrid라는 파일로 저장합니다. 

MFA 실행에 앞서 다음과 같은 과정이 필요합니다. 

1. wav-lab pair 생성

wav파일과 그 wav파일의 발화를 transcript한 lab파일이 필요합니다.

[processing_utils.ipynb](https://github.com/JH-lee95/Fastspeech2-Korean/blob/master/processing_utils.ipynb) 노트북 내의 audio_text_pair 함수를 실행하시면 됩니다.

해당 함수는 metadata로 부터 wav파일과 text를 인식하여, wav파일과 확장자만 다른 transcript파일(.lab) 을 생성합니다. 


![캡처1](https://user-images.githubusercontent.com/63226383/117935760-0568d500-b33f-11eb-857e-6024ed7a5421.PNG)

작업이 끝나면 위의 형태와 같이 wav-lab pair가 만들어져야 합니다.


2. lexicon 파일 생성 
 
가지고 있는 데이터셋 내의 모든 발화에 대한, phoneme을 기록한 lexicon 파일을 생성합니다.

[processing_utils.ipynb](https://github.com/JH-lee95/Fastspeech2-Korean/blob/master/processing_utils.ipynb) 노트북 내의 make_p_dict 와 make_lexicon 함수를 차례대로 실행해주세요.

![1](https://user-images.githubusercontent.com/63226383/117945618-7614ef00-b349-11eb-8e54-8d1a98bc0dab.PNG)

작업이 끝나면 위와 같은 형태를 띄는 p_lexicon.txt 파일이 만들어집니다. 


3. MFA 설치 

* MFA에 대한 자세한 설치 방법은 https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html 이곳을 확인해주세요.


4. MFA 실행

MFA의 경우 pre-trained된 한국어 acoustic model과 g2p 모델을 제공하고 있습니다. 하지만 해당 모델은 english phoneme을 생성하기 때문에 한국어 phoneme을 생성하기 위해서는 직접 train을 시켜주어야 합니다.

MFA 설치가 완료되었다면 아래와 같은 커멘드를 실행해주세요.

```
mfa train <데이터셋 위치> <p_lexicon의 위치> <out directory>
```

MFA가 정상적으로 실행되었을 경우 다음과 같은 형태의 TextGrid 파일이 만들어집니다.
![캡처](https://user-images.githubusercontent.com/63226383/117936797-3d244c80-b340-11eb-89d0-699f3499e8e8.PNG)



**(3) 데이터전처리**

1.hparms.py

- dataset : 데이터셋 폴더명
- data_path : dataset의 상위 폴더
- meta_name : metadata의 파일명 ex)transcript.v.1.4.txt
- textgrid_path : textgrid 압축 파일의 위치 (textgrid 파일들을 미리 압축해주세요)
- tetxgrid_name : textgird 압푹 파일의 파일명

2. preprocess.py

![캡처](https://user-images.githubusercontent.com/63226383/117941734-58458b00-b345-11eb-9fa8-47fc74c7a844.PNG)

해당 부분을 본인의 데이터셋 이름에 맞게 변경해주세요


3. data/kss.py

- line 19 : basename,text = parts[?],parts[?]  #각각 텍스트의 위치 ("|")로 split했을때, metadata에 기록된 wav와 text의 위치
- line 37 : basename,text = parts[?],parts[?]


위의 변경 작업이 모두 완료되면 아래의 커멘드를 실행해주세요.

```
python preprocess.py
```
    
# Train
모델 학습 전에, kss dataset에 대해 사전학습된 VocGAN(neural vocoder)을 [다운로드](https://drive.google.com/file/d/1GxaLlTrEhq0aXFvd_X1f4b-ev7-FH8RB/view?usp=sharing) 하여 ``vocoder/pretrained_models/`` 경로에 위치시킵니다.

다음으로, 아래의 커맨드를 입력하여 모델 학습을 수행합니다.
```
python train.py
```
학습된 모델은 ``ckpt/``에 저장되고 tensorboard log는 ``log/``에 저장됩니다. 학습시 evaluate 과정에서 생성된 음성은 ``eval/`` 폴더에 저장됩니다.

# Synthesis
학습된 파라미터를 기반으로 음성을 생성하는 명령어는 다음과 같습니다. 
```
python synthesis.py --step 500000
```
합성된 음성은  ```results/``` directory에서 확인하실 수 있습니다.

# Pretrained model
pretrained model(checkpoint)을 [다운로드](https://drive.google.com/file/d/1qkFuNLqPIm-A5mZZDPGK1mnp0_Lh00PN/view?usp=sharing)해 주세요.
그 후,  ```hparams.py```에 있는 ```checkpoint_path``` 변수에 기록된 경로에 위치시켜주시면 사전학습된 모델을 사용 가능합니다.


# Fine-Tuning
Pretrained model을 활용하여 Fine-tuning을 할 경우, 최소 30분 이상의 데이터가 권장됩니다. 10분 정도 분량의 데이터로 실험시 목소리와 발음은 대체적으로 비슷하게 따라하나 노이즈가 심했습니다.

Fine-tuning 시, Learning Rate의 조정이 필요합니다. Learning Rate는 적당히 낮은 값이 필요하며, 이는 경험적으로 알아내셔야 합니다. (저는 최종 step에서의 Learning Rate를 사용했습니다.)

```
python train.py --restore_step 350000 
```



# Tensorboard
```
tensorboard --logdir log/hp.dataset/
```
tensorboard log들은 ```log/hp.dataset/``` directory에 저장됩니다. 그러므로 위의 커멘드를 이용하여 tensorboard를 실행해 학습 상황을 모니터링 하실 수 있습니다.



# References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263), Y. Ren, *et al*.
- [ming024's FastSpeech2 impelmentation](https://github.com/ming024/FastSpeech2)
- [rishikksh20's VocGAN implementation](https://github.com/rishikksh20/VocGAN)
- [HGU-DLLAB](https://github.com/HGU-DLLAB/Korean-FastSpeech2-Pytorch)
- [TensorSpeech](https://github.com/TensorSpeech/TensorFlowTTS)
