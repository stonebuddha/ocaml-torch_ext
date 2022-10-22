(lang dune 3.4)

(name torch_ext)

(generate_opam_files true)

(maintainers "wayne.wangdi@yandex.com")

(package
 (name torch_ext)
 (synopsis "PyTorch bindings and probability distributions for OCaml")
 (description
  "The project provides some OCaml bindings for the PyTorch library (including\nsome probability distributions).")
 (depends
  ocaml
  dune
  (torch
   (= 0.14))))
