/*
 * ftsan.c -- Archer runtime library, TSan annotations for Fortran
 */
  
  //===----------------------------------------------------------------------===//
  //
  // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
  // See tools/archer/LICENSE.txt for details.
  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  //
  //===----------------------------------------------------------------------===//
  
  

void __attribute__((weak))
AnnotateHappensAfter(const char *file, int line, const volatile void *cv) {}
void __attribute__((weak))
AnnotateHappensBefore(const char *file, int line, const volatile void *cv) {}
void __attribute__((weak))
AnnotateIgnoreWritesBegin(const char *file, int line) {}
void __attribute__((weak)) AnnotateIgnoreWritesEnd(const char *file, int line) {
}

// This is a Fortran wrapper for TSan Annotation functions.

void annotatehappensafter_(const char *file, int *line,
                           const volatile void *cv) {
  AnnotateHappensAfter(file, *line, cv);
}
void annotatehappensbefore_(const char *file, int *line,
                            const volatile void *cv) {
  AnnotateHappensBefore(file, *line, cv);
}
void annotateignorewritesbegin_(const char *file, int line) {
  AnnotateIgnoreWritesBegin(file, line);
}
void annotateignorewritesend_(const char *file, int line) {
  AnnotateIgnoreWritesEnd(file, line);
}
