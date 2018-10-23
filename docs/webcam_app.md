* 사용법
  * cam [webcam_device#] [model_filename] [input_width] [input_height] [use_NNAPI]
  * [labels.txt](../models/labels.txt)와 [model file](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/models.md)이 현재 경로상에 있어야 함.
* 테스트 결과
  * inception과 mobilenet 두 개의 모델을 사용해본 결과, 깊은 모델일수록 GPU 사용 효과가 큰 듯 하다.
  * inception_v4 (input size = 299 * 299)
    * CPU: 약 0.25 fps
    * GPU: 약 0.45 fps
    * GPU(NNAPI)를 사용했을 때가 확실히 빠르다.
      ![inception_v4](./fig/inception_v4.png)
  * mobilenet_v2_1.0_224 (input size = 224 * 224)
    * GPU(NNAPI)를 사용했을 때 안정적인 성능을 보임.
    * CPU: 약 2.5 ~ 4.5 fps
      ![mobilenet_v2_1.0_224_CPU](./fig/mobilenet_v2_1.0_224_noNNAPI.png)
    * GPU: 약 3.85 fps
      ![mobilenet_v2_1.0_224_GPU](./fig/mobilenet_v2_1.0_224_NNAPI.png)
