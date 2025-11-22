# binpack_loader

PyO3 bindings that expose the Rust binpack reader (`sfbinpack`) as a Python module. The
extension currently implements the sparse HalfKP batch stream needed by
`halfkp/train.py`.

## Building locally

Install [maturin](https://github.com/PyO3/maturin) once (inside your Python environment):

```bash
python3 -m pip install maturin
```

Then build + install the module into the current environment:

```bash
cd binpack-rust-main/pybinpack
maturin develop --release
```

This command compiles the Rust crate as a `binpack_loader` Python extension and makes it
importable via `import binpack_loader`. Re-run the same command whenever you update the
Rust sources.
