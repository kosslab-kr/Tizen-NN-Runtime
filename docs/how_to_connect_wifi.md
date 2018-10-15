* Odroid XU4 + Tizen 5.0 환경
* download [wifi-manager package](http://download.tizen.org/snapshots/tizen/unified/tizen-unified_20181002.3/repos/standard/packages/armv7l/capi-network-wifi-manager-1.0.39-80.3.armv7l.rpm)
* install wifi-manager package
  * rpm -Uvh capi-network-wifi-manager-1.0.39-80.3.armv7l.rpm
* wifi 연결 (wifi_manager_test 명령어 사용)
  * `wifi_manager_test`
  * "Test Thread created..." 문구가 출력되면 엔터 입력
  * 숫자 혹은 알파벳을 입력하여 원하능 기능을 수행할 수 있다.
* 초기 연결 설정 : 1 -> 3 -> b -> c (AP 연결) -> a -> 0 순서로 
  * "1   - Wi-Fi init and set callbacks" 선택
  * "3   - Activate Wi-Fi device" 선택
  * "b   - Get AP list" 선택하면 아래와 같이 AP 리스트를 볼 수 있다.
    ```
    AP name : IPA2, state : Disconnected
    AP name : ollehWiFi , state : Disconnected
    AP name : IPA3, state : Disconnected
    AP name : ollehWiFi, state : Disconnected
    AP name : IPA4, state : Disconnected
    AP name : DTD, state : Disconnected
    AP name : IPA1, state : Disconnected
    AP name : KT_GiGA_2G_Wave2_E7D5, state : Disconnected
    AP name : SPCM, state : Disconnected
    AP name : KT_WiFi_2G_64E7, state : Disconnected
    AP name : KT_GiGA_2G_Wave2_135A, state : Disconnected
    AP name : SO070VOIP9E05, state : Disconnected
    AP name : miso24G, state : Disconnected
    ```
  * "c   - Connect" 선택 후 AP명, 암호 입력
  * "a   - Get Connected AP" 선택하면 연결된 AP 확인 가능
  * "0   - wifi_manager_test" 
* 이후에는 reboot 후에도 wifi가 자동적으로 활성화되며 기존에 연결했던 AP에 자동으로 연결된다.
