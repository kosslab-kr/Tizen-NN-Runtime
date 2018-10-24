* webcam app 개발
  * tflite를 사용한 object detection app 개발
    * webcam 연동
    * webcam 프레임을 model input 형식에 맞도록 변형
    * model output에 대한 label 로딩
    * model output의 top5 출력
    * float models 및 quantized models 모두 사용 가능하도록 함
    * mobilnet 및 inception 모델 테스트 및 NNAPI 사용 가능 확인
    * fps 계산 기능 개선
    * CMakeLists.txt 및 spec 파일 작성
  * 관련 문서 작성
    * webcam app 설명: webcam_app.md
    * webcam device 확인 방법: how_to_check_usb_webcam.md
    * webcam app 코드의 전반적인 설명: how_to_object_detection_using_webcam.md
* xor app packaging
  * 패키징 가능하도록 CMakeLists.txt 및 spec 파일 작성
  * 관련 문서 작성
    * xor 빌드 방법: how_to_build_xor.md
* tensorflow lite 관련
  * single build 방법 확인
  * 관련 문서 작성
    * tflite 싱글빌드 방법: how_to_build_tflite_only.md
* 그 외 tip 문서 작성
  * SDB 사용법: connect_target_with_sdb.md
  * odroid에서 wifi 사용법: how_to_connect_wifi.md
  * spec 파일 설명: spec_file_guide.md
