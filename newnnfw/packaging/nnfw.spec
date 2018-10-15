Name:    nnfw
Summary: nnfw
Version: 0.2
Release: 1
Group:   Development
License: Apache-2.0 and MIT

Source0: %{name}-%{version}.tar.gz
Source1: %{name}.manifest

%ifarch arm armv7l aarch64
BuildRequires:	cmake
BuildRequires:	python
BuildRequires:	boost-devel
BuildRequires:	gtest-devel
BuildRequires:	tensorflow-lite-devel

BuildRequires:	libarmcl-devel
%endif

Requires(post): /sbin/ldconfig
Requires(postun): /sbin/ldconfig

%description
nnfw is a high-performance, on-device neural network framework for Tizen

%{!?build_type: %define build_type Release}

%package test
Summary: NNFW Test
Requires: nnfw

%description test
NNFW test rpm. It does not depends on nnfw rpm since it contains nnfw runtime.

%define install_prefix /usr
%define test_install_prefix /opt/usr/nnfw-test

%ifarch %{arm}
%define target_arch armv7l
%endif
%ifarch x86_64
%define target_arch x86_64
%endif
%ifarch aarch64
%define target_arch aarch64
%endif

%{!?coverage_build: %define coverage_build 0}
%if %{coverage_build} == 1
%define build_options COVERAGE_BUILD=1 OBS_BUILD=1 BUILD_TYPE=Debug TARGET_ARCH=%{target_arch} TARGET_OS=tizen UPDATE_MODULE=0
%else
%define build_options OBS_BUILD=1 BUILD_TYPE=%{build_type} INSTALL_PATH=%{buildroot}%{install_prefix} TARGET_ARCH=%{target_arch} TARGET_OS=tizen UPDATE_MODULE=0
%endif

%prep
%setup -q
cp %{SOURCE1} .

%build
%ifarch arm armv7l aarch64
%{build_options} make %{?jobs:-j%jobs}
%endif

%install
%ifarch arm armv7l aarch64
%{build_options} make install

%if %{coverage_build} == 0
# nnfw-test rpm(like test-suite on cross build)
## install Product
mkdir -p %{buildroot}%{test_install_prefix}/Product/out
mv %{buildroot}%{install_prefix}/unittest %{buildroot}%{test_install_prefix}/Product/out
mv %{buildroot}%{install_prefix}/bin %{buildroot}%{test_install_prefix}/Product/out
cp -rf %{buildroot}%{install_prefix}/lib %{buildroot}%{test_install_prefix}/Product/out
rm -rf %{buildroot}%{install_prefix}/lib/pureacl
## install tests
cp -rf ./tests %{buildroot}%{test_install_prefix}/.
## install tools
mkdir -p %{buildroot}%{test_install_prefix}/tools
cp -rf ./tools/test_driver %{buildroot}%{test_install_prefix}/tools
%else
%{build_options} make build_coverage_suite
mkdir -p %{buildroot}%{test_install_prefix}
cp -rf Product/out/coverage-suite.tar.gz %{buildroot}%{test_install_prefix}/.
tar -zxf %{buildroot}%{test_install_prefix}/coverage-suite.tar.gz -C %{buildroot}%{test_install_prefix}
rm -rf %{buildroot}%{test_install_prefix}/coverage-suite.tar.gz
%endif
%endif

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig

%files
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l aarch64
%{install_prefix}/lib/*
%exclude %{install_prefix}/bin/*
%exclude %{install_prefix}/unittest/*
%endif

%files test
%manifest %{name}.manifest
%defattr(-,root,root,-)
%ifarch arm armv7l aarch64
%{test_install_prefix}/*
%endif

%changelog
* Thu Mar 15 2018 Chunseok Lee <chunseok.lee@samsung.com>
- Initial spec file for nnfw
