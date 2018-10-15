## GBS 빌드 환경 만들기

#### GBS 설치하기

1. Tizen Repository 등록

   ```
   $ cat /etc/apt/sources.list.d/tizen.list
   deb http://download.tizen.org/tools/latest-release/Ubuntu_16.04/ /
   ```

2. GBS 설치

   ```
   $ sudo apt-get update
   ...
   $ sudo apt-get install gbs
   ...
   ```

   * GBS 설치, 버전 확인

   ```
   $ gbs --version
   ```



#### 개발 환경 설정하기

1. GBS Build 환경 설정
   1. Tizen 사이트에서 ".gbs.conf" 내용 전체 복사

      https://source.tizen.org/documentation/developer-guide/environment-setup

   2. ".gbs.conf" 파일 수정

      ```
      $ sudo vim ~/.gbs.conf
      ```

   3. 전체 내용 삭제 후 복사해 온 내용 붙여넣기

   4. 필요한 내용 수정

      * 1, 2번 줄 내용 수정

      ```
      [general]
      profile =profile.unified_standard
      ```

      * 앞 빈칸 지우기
      * "buildconf = ~~" 삭제
      * 저장 후 vim 종료