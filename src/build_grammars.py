from tree_sitter import Language

Language.build_library(
  # Store the library in the `build` directory
  './resource/grammars/languages.so',

  # Include one or more languages
  [
    './resource/grammars/tree-sitter-c'
  ]
)