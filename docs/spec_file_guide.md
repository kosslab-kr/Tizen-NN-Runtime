* __RPM (Red Hat Package Manager)__
  * 리눅스에 사용되는 가장 일반적인 소프트웨어 패키징 방법
* __Spec File__
  * RPM 패키지를 생성하기 위한 레시피 파일
  * 패키지 설명정보와 스크립트를 담고 있음
* __Preamble__
  * Name
    * RPM 패키지의 이름을 설정한다.
  * Summary
    * RPM 패키지의 요약 정보
  * Version
    * 생성되는 패키지의 버전 정보를 설정한다. 여기에 설정된 버전 정보는 패키지 레이블 및 패키지 filename 에 설정되게 된다.
  * Release
    * Release 번호는 이 패키지가 현재 버전에서 몇번째 릴리즈 되었는지를 나타낸다. 패키지 레이블은 버전 정보와 릴리즈 번호가 합쳐져서 나타난 정보이다.
  * License
    * 라이센스 정보를 나타낸다. Copyright 와 비슷하다.
  * Group
    * Group은 해당 패키지가 어떤 종류의 소프트웨어 인지를 나타낸다.
  * Source
    * RPM을 만들기 위해 사용할 압축파일(*.tar.gz)를 지정한다. Source 뒤에 숫자를 붙여 여러 소스파일을 지정해 줄 수 있다. 숫자는 '0'부터 시작해야 한다.
  * Requires
    * 생성되는 RPM 의 설치 의존성을 설정한다. 
    * 기본적으로는 이름만 써도 되지만, 버전을 체크하도록 할 수 있다. 비교 연산자로는 (>, <, =, >=, <=)를 지원한다.
  * BuildRequires
    * RPM을 Build 하기 위해 필요한 RPM 패키지를 지정한다.
  * %description
    * 패키지에 대한 상세 설명을 설정한다. Summary 에서는 한줄로만 간단하게 패키지의 설명을 적을 수 있었지만 %description 에서는 하나 이상의 라인을 이용하여 더욱 많은 정보를 설정할 수 있다.
* __%prep__
  * %setup
    * source0에서 지정한 파일을 빌드 하기 위해 필요한 일들을 지정해준다. 압축을 푸는 작업이 주된 작업이다.
    * options
      * -q: 압축을 푸는 과정을 보여주지 않는다. (진행과정을 감춘다)
      * -n: 이름을 지정한다.
* __%build__
  * 프로그램 빌드를 위한 명령을 적어주는 부분이다.  실제로 어떤 매크로를 사용하는 것이 아닌 압축 풀린 소스를 가지고 원하는 소프트웨어를 빌드 하고, 그것을 패치하고 그 디렉토리로 이동하는 등의 명령을 주는 부분이다. 명령들이 쉘에 전달되는 또 다른 셋으로써, 어떠한 쉘 명령이든지(설명을 포함해서) 여기에 쓸 수 있다. 현재 작업 디렉토리는 각각의 단락마다 소스 디렉토리의 최상위 레벨 디렉토리로 리셋되므로 주의해야 한다.
* __설치와 제거의 선행/후행 스크립트__
  * 바이너리 패키지의 설치나 제거 전후에 실행할 스크립트를 기술 하는 부분이다. 주된 이유는 공유 라이브러리를 담고 있는 패키지를 설치하거나 제거하고 나서 ldconfig와 같은 명령을 실행하기 위해서이다.
  * %pre: 설치하기 전에 실행되는 스크립트  
  * %post: 설치한 후에 실행되는 스크립트  
  * %preun: 제거하기 전에 실행되는 스크립트
  * %postun: 제거한 후에 실행되는 스크립트
* __%install__
  * %build에서 컴파일된 프로그램을 설치하는 명령어를 적어주는 부분이다. 이것 역시 실제 어떠한 매크로가 아니다. 기본적으로 설치하는데 필요한 명령을 적어준다. make install에 쓰일 makefile을 패치하거나 make install을 여기서 할 수 있다. 또는 수동적인 쉘 명령으로 설치할 수도 있다. 현재 디렉토리가 소스 디렉토리의 가장 상위 디렉토리가 된다는 것을 주의 해야한다.
* __%files__
  * RPM으로 패키징될 파일들을 하나씩 지정해주는 부분이다. RPM은 make install의 결과로 어떠한 바이너리가 설치되는지 알 방법이 없다. 따라서 빌드에 포함될 대상들을 지정해 주어야 한다. 파일을 지정할 때 $RPM_BUILD_ROOT가 지정하는 디렉토리를 기점으로 지정해 주어야 한다.
* __%defattr__
  * 기본 권한을 설정
  * %defattr(\<file_mode\>, \<user\>, \<group\>, \<dir_mode\>)
* __%package__
  * 서브-패키지를 생성하기 위해 사용된다.
  * Name은 'tensorflow'이고 '%package devel'이라고 지정한다면 'tensorflow-devel.rpm'이란 이름의 파일이 생성된다.
* __리눅스 명령어__
  * pushd, popd
    * 디렉토리경로를 스택에 보관(pushd)하고 인출(popd)하는 리눅스 명령어
    * pushd 뒤에 이동하려는 목적지경로를 넣으면 현재 위치를 스택에 보관하고 해당위치로 이동함
    * popd를 하면 pushd를 했던 곳으로 복귀함
  * cat
    * 리다이렉션을 통해 입출력 방향을 지정해줄 수 있다.
    * cat << EOF > ./git
      * EOF가 나올 때까지 표준입력을 받아 현재 경로 내 'git' 파일을 생성하여 입력 받은 내용을 쓴다.
    * cat file file2 file3 > file4
      *  해당 파일 세 개의 파일을 모두 합쳐서 새로운 file4로 만들어 주는 것이다. file, file1, file2의 내용은 기존 내용과 달라지지 않는다.
  * mkdir -p
    * 상위 경로까지 함께 생성
  * chmod
    * options
      * a+x: 모든 사용자(a)의 실행(x) 권한 추가(+)
      * a-wx: 모든 사용자(a)의 쓰기(w), 실행(x) 권한 제거(-)
  * export PATH=${PATH}:\`pwd\`
    * 현재 경로를 환경변수에 추가
  * tar
    * options
      * x: 묶음을 해제	  
      * c: 파일을 묶음
      * v: 묶음/해제 과정을 화면에 표시
      * z: gunzip을 사용
      * f: 파일 이름을 지정
      * p: 권한(permission)을 원본과 동일하게 유지
  * sed
    * 스트림 에디터라 부르며 지정한 지시에 따라 파일이나 파이프라인 입력을 편집해서 출력해주는 명령어
    * [참고](http://www.incodom.kr/Linux/%EA%B8%B0%EB%B3%B8%EB%AA%85%EB%A0%B9%EC%96%B4/sed)
  * make -f <file>
    * file을 makefile로 읽는다.
  * install
    * copy files and set attributes
	* options
	  * -m 권한을 지정하여 복사
  * ldconfig
    * ldconfig는 runtime때 여러 위치에 존재하는 shared object, 쉽게말해 동적 라이브러리를 연결해주는 dynamic linker를 설정하는 command이다.
	* so 파일들의 경로는 /etc/ld.so.conf 파일에 지정되어 있음