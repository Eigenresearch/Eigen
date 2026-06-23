; ModuleID = 'EigenLLVMModule'
source_filename = "eigen_source.eig"

%Qubit = type opaque
%Result = type opaque

define void @main() #0 {
entry:
  br label %B0

B0:
  br label %B9

B1:

B2:

B3:
  call void @__quantum__qis__x__body(%Qubit* %y)

B4:

B5:
  call void @__quantum__qis__cnot__body(%Qubit* %x, %Qubit* %y)

B6:

B7:
  call void @__quantum__qis__x__body(%Qubit* %x)
  call void @__quantum__qis__cnot__body(%Qubit* %x, %Qubit* %y)
  call void @__quantum__qis__x__body(%Qubit* %x)

B8:

B9:
  %q0 = call %Qubit* @__quantum__rt__qubit_allocate()
  %q1 = call %Qubit* @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__x__body(%Qubit* %q1)
  call void @__quantum__qis__h__body(%Qubit* %q0)
  call void @__quantum__qis__h__body(%Qubit* %q1)

B10:
  call void @__quantum__qis__h__body(%Qubit* %q0)
  %result_1 = select i1 true, i32 None, i32 None
  %res_result_2 = call %Result* @__quantum__qis__m__body(%Qubit* %q0)
  %result_2 = call i1 @__quantum__rt__result_get_one(%Result* %res_result_2)
  call void @print_bool(i1 %result_2)
  %t1 = icmp eq i1 %result_2, 1
  %t2 = icmp eq i1 %t1, 1
  br i1 %t2, label %B12, label %B11

B11:

B12:
  ret void

}

declare %Qubit* @__quantum__rt__qubit_allocate()
declare void @__quantum__rt__qubit_release(%Qubit*)
declare void @__quantum__qis__h__body(%Qubit*)
declare void @__quantum__qis__x__body(%Qubit*)
declare void @__quantum__qis__y__body(%Qubit*)
declare void @__quantum__qis__z__body(%Qubit*)
declare void @__quantum__qis__s__body(%Qubit*)
declare void @__quantum__qis__t__body(%Qubit*)
declare void @__quantum__qis__rx__body(double, %Qubit*)
declare void @__quantum__qis__ry__body(double, %Qubit*)
declare void @__quantum__qis__rz__body(double, %Qubit*)
declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*)
declare void @__quantum__qis__cz__body(%Qubit*, %Qubit*)
declare %Result* @__quantum__qis__m__body(%Qubit*)
declare i1 @__quantum__rt__result_get_one(%Result*)
declare void @print_int(i32)
declare void @print_double(double)
declare void @print_bool(i1)