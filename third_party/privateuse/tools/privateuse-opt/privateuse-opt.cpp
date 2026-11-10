#include "RegisterPrivateUseDialects.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char** argv)
{
  mlir::DialectRegistry registry;
  registerPrivateUseDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "PrivateUse test driver\n", registry));
}
