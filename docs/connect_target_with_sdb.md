* [wifi 연결 참고](https://github.com/kosslab-kr/Tizen-NN-Runtime/blob/master/docs/how_to_connect_wifi.md)
* sdb 사용을 위해 tizen studio를 설치한다.
  * [download tizen-studio](https://developer.tizen.org/development/tizen-studio/download)
  * [installing-tizen-studio](https://developer.tizen.org/development/tizen-studio/download/installing-tizen-studio)
  * 설치하면 tizen-studio/tools에 sdb가 존재한다.
* sdb 가 있는 폴더를 PATH 에 등록
  * sudo gedit /etc/environment
  * environment 파일 내용 수정 : PATH="/home/<ID>/tizen-studio/tools/:~~~<기존 PATH>
  * PATH 업데이트 : sudo source /etc/environment
* host와 target(odroid)가 동일 AP를 사용하는 상태에서,
* target 접속
  * sdb connect [target_ip]
* 연결 확인
  * sdb devices
* target shell 접속
  * sdb shell
* target shell에 들어가지 않고 명령을 실행할 수 있다.
  * sdb shell [command]
  * 예> sdb shell mount -o rw,remount /
* 파일 전송
  * sdb root on
  * sdb shell mount -o rw,remount /
  * sdb push [file_name] [target_path]
