Name: cvcam
Version: 0.0.1
Release: 1
License: Apache-2.0
Summary: opencv webcam application
Packager: lazineer@gmail.com
Group: Application
Source: %{name}-%{version}.tar.gz
BuildRequires: cmake

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
* Sun Oct 09 2018 lazineer@gmail.com
 - initial release

