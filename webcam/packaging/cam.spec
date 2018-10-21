Name: cam
Version: 1.0.0
Release: 1
License: Apache-2.0
Summary: opencv webcam application
Packager: lazineer@gmail.com
Group: Application
Source: %{name}-%{version}.tar.gz
BuildRequires: cmake
#BuildRequires: gtk+-2.0
BuildRequires: pkgconfig(opencv)
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
* Sun Oct 09 2018 lazineer@gmail.com
 - initial release

