fn main() {
    // LMDB uses Windows security APIs (InitializeSecurityDescriptor,
    // SetSecurityDescriptorDacl) for lock file setup.
    println!("cargo:rustc-link-lib=advapi32");
}
