## Commands

```
// with simd
export RUSTFLAGS="-C embed-bitcode=yes -C codegen-units=1 -C opt-level=3 -Ctarget-feature=+simd128 --cfg web_sys_unstable_apis" & wasm-pack build --release --out-dir pkg/simd --target web
// without simd
export RUSTFLAGS="-C embed-bitcode=yes -C codegen-units=1 -C opt-level=3 --cfg web_sys_unstable_apis" & wasm-pack build --release --out-dir pkg/no_simd --target web
```
