// RUN: privateuse-opt %s -privateuse-puc-launch-kernel | FileCheck %s

// The kernel entry keeps its own calling convention but is marked GLOBAL,
// while a generated <kernel>_launcher wrapper owns the extern "C" symbol and
// forwards its arguments with a plain puc.launch. Private helpers that are
// called in the module are not wrapped.

// CHECK: puc.func private @helper(!pumem.ptr<f32, 0>, index) -> index

puc.func private @helper(!pumem.ptr<f32, 0>, index) -> index

// CHECK: puc.func @memory(%[[SRC:.*]]: !pumem.ptr<f32, 0>, %[[N:.*]]: index) attributes {pumem.outlined = "GLOBAL"} {
// CHECK:   {{.*}} = puc.call @helper(%[[SRC]], %[[N]]) : (!pumem.ptr<f32, 0>, index) -> index
// CHECK:   puc.return
// CHECK: }

puc.func @memory(%src: !pumem.ptr<f32, 0>, %n: index) {
  %r = puc.call @helper(%src, %n) : (!pumem.ptr<f32, 0>, index) -> index
  puc.return
}

// CHECK: puc.func @memory_launcher(%[[LSRC:.*]]: !pumem.ptr<f32, -1>, %[[LN:.*]]: index) attributes {pumem.outlined = "extern {{.*}}C{{.*}}"} {
// CHECK:   puc.launch @memory(%[[LSRC]], %[[LN]]) : !pumem.ptr<f32, -1>, index
// CHECK:   puc.return
// CHECK: }
