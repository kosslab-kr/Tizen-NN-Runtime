- tensorflow-lite single build 지원 (2018-08-27)
  - Full build takes too much time on developer's local machine. This commit allows user to build tensorflow-lite package w/o tensorflow build
  - [상세 이력 보기](https://git.tizen.org/cgit/platform/upstream/tensorflow/commit/?h=tizen&id=1d41ec19c82cbf098c91d688f47b1ef191506390)
- tensorflow.spec
  - tflite_single_build 값을 통해 전부 빌드할지 lite만 빌드할지 결정
  - 아래와 같이 초기값 0으로 정의됨
    `%{!?tflite_single_build: %define tflite_single_build 0}`
  - 아래와 같이 tflite_single_build가 0이면 모두 빌드하도록 되어 있음
    ```
    %if "%{tflite_single_build}" == "0"
    ...
    %endif
    ```
  - 따라서 빌드 시 0 이외의 값으로 지정해주면 lite만 빌드함
- define 옵션으로 tflite_single_build 값 지정하여 빌드
  `gbs build --include-all -A armv7l --define 'tflite_single_build 1'`
