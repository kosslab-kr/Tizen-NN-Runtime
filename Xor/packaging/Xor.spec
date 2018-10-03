Name: org.tizen.Xor
Version: 0.0.1
Release: 1
License: Apache-2.0
Summary: tflite xor application
Packager: Author <E-mail>
Group: Application
Source: %{name}-%{version}.tar.gz
BuildRequires: cmake
BuildRequires: tensorflow-lite-devel
%description
Platform Project

%prep
%setup -q

%build
cmake . \
	-DCMAKE_BUILD_TYPE=%{BUILD_TYPE}
make

%install
make install DESTDIR=$RPM_BUILD_ROOT/usr

%clean
make clean

%files
%defattr(644, root, root)
%attr(755, root, root) /usr/bin

%changelog
* Sat Mar 24 2012 Author <E-mail>
 - initial release
