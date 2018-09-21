* 개발자 사이트(https://source.tizen.org)

* rpm 파일 및 이미지파일 다운로드(https://download.tizen.org)
  * relaeses 또는 snapshots 폴더, 이미지랑 GBS 빌드를 위한 repo등
  * Tizen 4.0 : /snapshots/tizen/4.0-unified
  * Tizen 5.0 : /snapshots/tizen/unified

* Tizen에서 사용되는 프로젝트 git(https://review.tizen.org)
  * repository clone을 위해서 회원가입 및 ssh설정 필요(https://source.tizen.org/documentation/developer-guide/environment-setup)

  * tizen git 참고 프로젝트 리스트(origin/tizen 또는 origin/upstream branch확인)
    * nnfw : https://git.tizen.org/cgit/platform/core/ml/nnfw
             https://review.tizen.org/gerrit/#/admin/projects/platform/core/ml/nnfw
    * tensorflow : https://git.tizen.org/cgit/platform/upstream/tensorflow
             https://review.tizen.org/gerrit/#/admin/projects/platform/upstream/tensorflow
    * armcl : https://git.tizen.org/cgit/platform/upstream/armcl
             https://review.tizen.org/gerrit/#/admin/projects/platform/upstream/armcl

  * gbs 빌드 실행 : ex)gbs build --include-all --arch armv7l

* TensorFlow Lite(https://www.tensorflow.org/mobile/tflite/)
  * tflite 예제(https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/examples)
  * [tensorflow lite 설명](https://jaehwant.github.io/machinelearning/2018/01/04/9/)
  
* TensorFlow Lite Optimizing Converter
  * 학습된 tensorflow model을 tensorflow lite에서 사용하기 위한 컨버터
  * [toco github page](http://blog.canapio.com/tag/FlatBuffer)
  * http://gmground.tistory.com/14

* Arm Compute Library(https://developer.arm.com/technologies/compute-library)
  * armCL예제(https://github.com/ARM-software/ComputeLibrary/tree/master/examples)

* Arm NN(https://github.com/ARM-software/armnn)

* nnfw 관련
  * [Tizen 5.0 Public M1 release note](https://developer.tizen.org/tizen/release-notes/tizen-5.0-public-m1)
    * 2018.05.31 릴리즈 (Experimental Release)
    * ACL 기반의 CPU/GPU 가속 지원
    * Android NN API 일부 호환
    * TensorFlow Lite 일부 호환
    * Inception V3 모델 지원
  * nnfw README.md 내용
    * 고성능의 on-device 신경망 추론 프레임워크
    * 타이젠 또는 Smart Machine Platform(SMP)과 같은 타겟 플랫폼에서 CPU/GPU/NPU 등의 프로세스상에 주어진 NN 모델의 추론을 수행하는 고성능 on-device 신경망 제공
    * experimental 버전으로서 Inception V3만 실행할 수 있는 제한된 기능만 제공
    * 향후 릴리즈에서 이전 버전과의 호환성이 보장되지 않을 수 있음
  * Inception V3
    * 2014년 ILSVRC(ImageNet Large Scale Visual Recognition Challenge)에서 1등을 차지한 GoogLeNet(Inception V1)의 개량 모델
    * V2에서 3x3 conv만 사용하도록 변경했으며 V3에서 여러가지 성능향상을 위한 기법을 적용함
    * 구성도
    ![Inception V3](https://cloud.google.com/tpu/docs/images/inceptionv3onc--oview.png)
