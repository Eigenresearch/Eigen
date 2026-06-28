; ModuleID = "EigenLLVMModule"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare i8* @"__quantum__rt__qubit_allocate"()

declare void @"__quantum__rt__qubit_release"(i8* %".1")

declare void @"__quantum__qis__h__body"(i8* %".1")

declare void @"__quantum__qis__x__body"(i8* %".1")

declare void @"__quantum__qis__y__body"(i8* %".1")

declare void @"__quantum__qis__z__body"(i8* %".1")

declare void @"__quantum__qis__s__body"(i8* %".1")

declare void @"__quantum__qis__t__body"(i8* %".1")

declare void @"__quantum__qis__rx__body"(double %".1", i8* %".2")

declare void @"__quantum__qis__ry__body"(double %".1", i8* %".2")

declare void @"__quantum__qis__rz__body"(double %".1", i8* %".2")

declare void @"__quantum__qis__cnot__body"(i8* %".1", i8* %".2")

declare void @"__quantum__qis__cz__body"(i8* %".1", i8* %".2")

declare void @"__quantum__qis__swap__body"(i8* %".1", i8* %".2")

declare i8* @"__quantum__qis__mz__body"(i8* %".1")

declare i8* @"__quantum__rt__result_get_one"()

declare i1 @"__quantum__rt__result_equal"(i8* %".1", i8* %".2")

declare void @"eigen_qrt_print_int"(i64 %".1")

declare void @"eigen_qrt_print_bool"(i1 %".1")

declare void @"eigen_qrt_print_float"(double %".1")

declare void @"eigen_qrt_print_string"(i8* %".1")

declare void @"eigen_qrt_panic_div_zero"()

declare void @"llvm.trap"()

@"global_sim" = global i8* null
define i32 @"main"()
{
entry:
  %"c0" = alloca i1
  %"c1" = alloca i1
  %"q0" = alloca i8*
  %"q1" = alloca i8*
  br label %"B0"
B0:
  br label %"B3"
B3:
  %".4" = call i8* @"__quantum__rt__qubit_allocate"()
  store i8* %".4", i8** %"q0"
  %".6" = call i8* @"__quantum__rt__qubit_allocate"()
  store i8* %".6", i8** %"q1"
  %".8" = load i8*, i8** %"q0"
  call void @"__quantum__qis__h__body"(i8* %".8")
  %".10" = load i8*, i8** %"q1"
  call void @"__quantum__qis__h__body"(i8* %".10")
  %".12" = load i8*, i8** %"q0"
  %".13" = load i8*, i8** %"q1"
  call void @"__quantum__qis__cz__body"(i8* %".12", i8* %".13")
  %".15" = load i8*, i8** %"q0"
  %".16" = load i8*, i8** %"q1"
  call void @"diffuse_2"(i8* %".15", i8* %".16")
  %".18" = trunc i64 0 to i1
  store i1 %".18", i1* %"c0"
  %".20" = trunc i64 0 to i1
  store i1 %".20", i1* %"c1"
  %".22" = load i8*, i8** %"q0"
  %".23" = call i8* @"__quantum__qis__mz__body"(i8* %".22")
  %".24" = call i8* @"__quantum__rt__result_get_one"()
  %".25" = call i1 @"__quantum__rt__result_equal"(i8* %".23", i8* %".24")
  store i1 %".25", i1* %"c0"
  %".27" = load i8*, i8** %"q1"
  %".28" = call i8* @"__quantum__qis__mz__body"(i8* %".27")
  %".29" = call i8* @"__quantum__rt__result_get_one"()
  %".30" = call i1 @"__quantum__rt__result_equal"(i8* %".28", i8* %".29")
  store i1 %".30", i1* %"c1"
  %".32" = load i1, i1* %"c0"
  call void @"eigen_qrt_print_bool"(i1 %".32")
  %".34" = load i1, i1* %"c1"
  call void @"eigen_qrt_print_bool"(i1 %".34")
  %".36" = load i1, i1* %"c0"
  %".37" = zext i1 %".36" to i64
  %".38" = icmp eq i64 %".37", 1
  %".39" = icmp eq i1 %".38", 1
  br i1 %".39", label %"B5", label %"B4"
B4:
  %".41" = getelementptr inbounds [97 x i8], [97 x i8]* @".str0", i32 0, i32 0
  br label %"B5"
B5:
  %".43" = load i1, i1* %"c1"
  %".44" = zext i1 %".43" to i64
  %".45" = icmp eq i64 %".44", 1
  %".46" = icmp eq i1 %".45", 1
  br i1 %".46", label %"B7", label %"B6"
B6:
  %".48" = getelementptr inbounds [97 x i8], [97 x i8]* @".str1", i32 0, i32 0
  br label %"B7"
B7:
  ret i32 0
}

define void @"diffuse_2"(i8* %"q0.1", i8* %"q1.1")
{
entry:
  %"q0" = alloca i8*
  %"q1" = alloca i8*
  store i8* %"q0.1", i8** %"q0"
  store i8* %"q1.1", i8** %"q1"
  br label %"B1"
B1:
  %".7" = load i8*, i8** %"q0"
  call void @"__quantum__qis__h__body"(i8* %".7")
  %".9" = load i8*, i8** %"q1"
  call void @"__quantum__qis__h__body"(i8* %".9")
  %".11" = load i8*, i8** %"q0"
  call void @"__quantum__qis__x__body"(i8* %".11")
  %".13" = load i8*, i8** %"q1"
  call void @"__quantum__qis__x__body"(i8* %".13")
  %".15" = load i8*, i8** %"q0"
  %".16" = load i8*, i8** %"q1"
  call void @"__quantum__qis__cz__body"(i8* %".15", i8* %".16")
  %".18" = load i8*, i8** %"q0"
  call void @"__quantum__qis__x__body"(i8* %".18")
  %".20" = load i8*, i8** %"q1"
  call void @"__quantum__qis__x__body"(i8* %".20")
  %".22" = load i8*, i8** %"q0"
  call void @"__quantum__qis__h__body"(i8* %".22")
  %".24" = load i8*, i8** %"q1"
  call void @"__quantum__qis__h__body"(i8* %".24")
  ret void
B2:
  ret void
}

@".str0" = internal constant [97 x i8] c"Assertion Failed: BinaryOpNode(VarRefNode(c0) == LiteralNode(1: int)) == LiteralNode(True: bool)\00"
@".str1" = internal constant [97 x i8] c"Assertion Failed: BinaryOpNode(VarRefNode(c1) == LiteralNode(1: int)) == LiteralNode(True: bool)\00"