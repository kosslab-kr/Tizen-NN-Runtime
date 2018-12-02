## GBS로 Tensorflow Lite 빌드 후 장치에 설치하기

#### Tensorflow Lite 소스코드 다운로드

1. tizen.review.org 가입 후 settings -> ssh 등록

2. 소스코드 clone 하기

```
$ git clone https://review.tizen.org/gerrit/#/admin/projects/platform/upstream/tensorflow
```

3. 브랜치 확인 후 checkout, 폴더 이동

####  소스코드 빌드

1. 빌드하기

   * Tizen 5.0 

   ```
   $ gbs build –A armv7l –P unified_standard
   ```

   - Tizen 4.0

   ```
   $ gbs build –A armv7l –P 4.0-unified_standard
   ```

   - Single build시 추가해야 할 옵션

   ```
   --define 'tflite_single_build 1'
   ```

2. rpm 파일 확인하기

   ```
   $ ll ~/GBS_ROOT/local/repos/unified_standard/armv7l/RPMS/
   ```

#### rpm 파일 디바이스 설치

1. sdb tool 사용 가능 여부 확인

   - "sdb" 명령어 실행되지 않는 경우
     - Tizen Studio 2.5 설치 여부 확인
     - "/home/<PC 계정>/tizen-studio/tools:" 를 PATH에 추가 : "/etc/environment" 파일 수정

2. rpm 폴더 이동

   ```
   $ cd ~/GBS-ROOT/local/repos/unified_standard/armv7l/RPMS
   ```

3. rpm 파일을 Tizen 디바이스에 설치

   1. PC - Tizen 디바이스 연결

   ```
   $ sdb root on
   ```

   2. Tizen 디바이스 쓰기 권한 얻기

   ```
   $ sdb shell mount -o remount,rw /
   ```

   3. rpm 파일을 디바이스 root 폴더에 넣기

   ```
   $ sdb push <rpm 파일명> /root
   ```

   4. rpm 파일 설치

   ```
   $ sdb shell rpm -Uvh <rpm 파일명>
   ```

   5. rpm 파일 삭제

   ```
   $ sdb shell rm -rf <rpm 파일명>
   ```
