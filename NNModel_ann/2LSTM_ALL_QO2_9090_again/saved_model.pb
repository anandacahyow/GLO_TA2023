ял!
ђ¤
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
э
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements(
handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
ѕ
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ
ћ
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
ѕ"serve*2.11.02v2.11.0-rc2-15-g6290819256d8Ще
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
~
Adam/v/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_4/bias
w
'Adam/v/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_4/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_4/bias
w
'Adam/m/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_4/bias*
_output_shapes
:*
dtype0
є
Adam/v/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*&
shared_nameAdam/v/dense_4/kernel

)Adam/v/dense_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_4/kernel*
_output_shapes

:Z*
dtype0
є
Adam/m/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*&
shared_nameAdam/m/dense_4/kernel

)Adam/m/dense_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_4/kernel*
_output_shapes

:Z*
dtype0
Ў
 Adam/v/lstm_10/lstm_cell_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:У*1
shared_name" Adam/v/lstm_10/lstm_cell_10/bias
њ
4Adam/v/lstm_10/lstm_cell_10/bias/Read/ReadVariableOpReadVariableOp Adam/v/lstm_10/lstm_cell_10/bias*
_output_shapes	
:У*
dtype0
Ў
 Adam/m/lstm_10/lstm_cell_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:У*1
shared_name" Adam/m/lstm_10/lstm_cell_10/bias
њ
4Adam/m/lstm_10/lstm_cell_10/bias/Read/ReadVariableOpReadVariableOp Adam/m/lstm_10/lstm_cell_10/bias*
_output_shapes	
:У*
dtype0
х
,Adam/v/lstm_10/lstm_cell_10/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ZУ*=
shared_name.,Adam/v/lstm_10/lstm_cell_10/recurrent_kernel
«
@Adam/v/lstm_10/lstm_cell_10/recurrent_kernel/Read/ReadVariableOpReadVariableOp,Adam/v/lstm_10/lstm_cell_10/recurrent_kernel*
_output_shapes
:	ZУ*
dtype0
х
,Adam/m/lstm_10/lstm_cell_10/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ZУ*=
shared_name.,Adam/m/lstm_10/lstm_cell_10/recurrent_kernel
«
@Adam/m/lstm_10/lstm_cell_10/recurrent_kernel/Read/ReadVariableOpReadVariableOp,Adam/m/lstm_10/lstm_cell_10/recurrent_kernel*
_output_shapes
:	ZУ*
dtype0
А
"Adam/v/lstm_10/lstm_cell_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ZУ*3
shared_name$"Adam/v/lstm_10/lstm_cell_10/kernel
џ
6Adam/v/lstm_10/lstm_cell_10/kernel/Read/ReadVariableOpReadVariableOp"Adam/v/lstm_10/lstm_cell_10/kernel*
_output_shapes
:	ZУ*
dtype0
А
"Adam/m/lstm_10/lstm_cell_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ZУ*3
shared_name$"Adam/m/lstm_10/lstm_cell_10/kernel
џ
6Adam/m/lstm_10/lstm_cell_10/kernel/Read/ReadVariableOpReadVariableOp"Adam/m/lstm_10/lstm_cell_10/kernel*
_output_shapes
:	ZУ*
dtype0
Ћ
Adam/v/lstm_9/lstm_cell_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:У*/
shared_name Adam/v/lstm_9/lstm_cell_9/bias
ј
2Adam/v/lstm_9/lstm_cell_9/bias/Read/ReadVariableOpReadVariableOpAdam/v/lstm_9/lstm_cell_9/bias*
_output_shapes	
:У*
dtype0
Ћ
Adam/m/lstm_9/lstm_cell_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:У*/
shared_name Adam/m/lstm_9/lstm_cell_9/bias
ј
2Adam/m/lstm_9/lstm_cell_9/bias/Read/ReadVariableOpReadVariableOpAdam/m/lstm_9/lstm_cell_9/bias*
_output_shapes	
:У*
dtype0
▒
*Adam/v/lstm_9/lstm_cell_9/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ZУ*;
shared_name,*Adam/v/lstm_9/lstm_cell_9/recurrent_kernel
ф
>Adam/v/lstm_9/lstm_cell_9/recurrent_kernel/Read/ReadVariableOpReadVariableOp*Adam/v/lstm_9/lstm_cell_9/recurrent_kernel*
_output_shapes
:	ZУ*
dtype0
▒
*Adam/m/lstm_9/lstm_cell_9/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ZУ*;
shared_name,*Adam/m/lstm_9/lstm_cell_9/recurrent_kernel
ф
>Adam/m/lstm_9/lstm_cell_9/recurrent_kernel/Read/ReadVariableOpReadVariableOp*Adam/m/lstm_9/lstm_cell_9/recurrent_kernel*
_output_shapes
:	ZУ*
dtype0
Ю
 Adam/v/lstm_9/lstm_cell_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	У*1
shared_name" Adam/v/lstm_9/lstm_cell_9/kernel
ќ
4Adam/v/lstm_9/lstm_cell_9/kernel/Read/ReadVariableOpReadVariableOp Adam/v/lstm_9/lstm_cell_9/kernel*
_output_shapes
:	У*
dtype0
Ю
 Adam/m/lstm_9/lstm_cell_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	У*1
shared_name" Adam/m/lstm_9/lstm_cell_9/kernel
ќ
4Adam/m/lstm_9/lstm_cell_9/kernel/Read/ReadVariableOpReadVariableOp Adam/m/lstm_9/lstm_cell_9/kernel*
_output_shapes
:	У*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
І
lstm_10/lstm_cell_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:У**
shared_namelstm_10/lstm_cell_10/bias
ё
-lstm_10/lstm_cell_10/bias/Read/ReadVariableOpReadVariableOplstm_10/lstm_cell_10/bias*
_output_shapes	
:У*
dtype0
Д
%lstm_10/lstm_cell_10/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ZУ*6
shared_name'%lstm_10/lstm_cell_10/recurrent_kernel
а
9lstm_10/lstm_cell_10/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_10/lstm_cell_10/recurrent_kernel*
_output_shapes
:	ZУ*
dtype0
Њ
lstm_10/lstm_cell_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ZУ*,
shared_namelstm_10/lstm_cell_10/kernel
ї
/lstm_10/lstm_cell_10/kernel/Read/ReadVariableOpReadVariableOplstm_10/lstm_cell_10/kernel*
_output_shapes
:	ZУ*
dtype0
Є
lstm_9/lstm_cell_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:У*(
shared_namelstm_9/lstm_cell_9/bias
ђ
+lstm_9/lstm_cell_9/bias/Read/ReadVariableOpReadVariableOplstm_9/lstm_cell_9/bias*
_output_shapes	
:У*
dtype0
Б
#lstm_9/lstm_cell_9/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ZУ*4
shared_name%#lstm_9/lstm_cell_9/recurrent_kernel
ю
7lstm_9/lstm_cell_9/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_9/lstm_cell_9/recurrent_kernel*
_output_shapes
:	ZУ*
dtype0
Ј
lstm_9/lstm_cell_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	У**
shared_namelstm_9/lstm_cell_9/kernel
ѕ
-lstm_9/lstm_cell_9/kernel/Read/ReadVariableOpReadVariableOplstm_9/lstm_cell_9/kernel*
_output_shapes
:	У*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:Z*
dtype0
Є
serving_default_lstm_9_inputPlaceholder*+
_output_shapes
:         *
dtype0* 
shape:         
Ю
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_9_inputlstm_9/lstm_cell_9/kernel#lstm_9/lstm_cell_9/recurrent_kernellstm_9/lstm_cell_9/biaslstm_10/lstm_cell_10/kernel%lstm_10/lstm_cell_10/recurrent_kernellstm_10/lstm_cell_10/biasdense_4/kerneldense_4/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *-
f(R&
$__inference_signature_wrapper_199044

NoOpNoOp
ЊD
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╬C
value─CB┴C B║C
╬
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
┴
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
┴
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
Ц
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&_random_generator* 
д
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias*
<
/0
01
12
23
34
45
-6
.7*
<
/0
01
12
23
34
45
-6
.7*
* 
░
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
:trace_0
;trace_1
<trace_2
=trace_3* 
6
>trace_0
?trace_1
@trace_2
Atrace_3* 
* 
Ђ
B
_variables
C_iterations
D_learning_rate
E_index_dict
F
_momentums
G_velocities
H_update_step_xla*

Iserving_default* 

/0
01
12*

/0
01
12*
* 
Ъ

Jstates
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ptrace_0
Qtrace_1
Rtrace_2
Strace_3* 
6
Ttrace_0
Utrace_1
Vtrace_2
Wtrace_3* 
* 
с
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^_random_generator
_
state_size

/kernel
0recurrent_kernel
1bias*
* 

20
31
42*

20
31
42*
* 
Ъ

`states
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
ftrace_0
gtrace_1
htrace_2
itrace_3* 
6
jtrace_0
ktrace_1
ltrace_2
mtrace_3* 
* 
с
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
t_random_generator
u
state_size

2kernel
3recurrent_kernel
4bias*
* 
* 
* 
* 
Љ
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 

{trace_0
|trace_1* 

}trace_0
~trace_1* 
* 

-0
.1*

-0
.1*
* 
Ќ
non_trainable_variables
ђlayers
Ђmetrics
 ѓlayer_regularization_losses
Ѓlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

ёtrace_0* 

Ёtrace_0* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_9/lstm_cell_9/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#lstm_9/lstm_cell_9/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUElstm_9/lstm_cell_9/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_10/lstm_cell_10/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_10/lstm_cell_10/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_10/lstm_cell_10/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

є0
Є1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
њ
C0
ѕ1
Ѕ2
і3
І4
ї5
Ї6
ј7
Ј8
љ9
Љ10
њ11
Њ12
ћ13
Ћ14
ќ15
Ќ16*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
D
ѕ0
і1
ї2
ј3
љ4
њ5
ћ6
ќ7*
D
Ѕ0
І1
Ї2
Ј3
Љ4
Њ5
Ћ6
Ќ7*
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

/0
01
12*

/0
01
12*
* 
ў
ўnon_trainable_variables
Ўlayers
џmetrics
 Џlayer_regularization_losses
юlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

Юtrace_0
ъtrace_1* 

Ъtrace_0
аtrace_1* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

20
31
42*

20
31
42*
* 
ў
Аnon_trainable_variables
бlayers
Бmetrics
 цlayer_regularization_losses
Цlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

дtrace_0
Дtrace_1* 

еtrace_0
Еtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
ф	variables
Ф	keras_api

гtotal

Гcount*
M
«	variables
»	keras_api

░total

▒count
▓
_fn_kwargs*
ke
VARIABLE_VALUE Adam/m/lstm_9/lstm_cell_9/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/lstm_9/lstm_cell_9/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/m/lstm_9/lstm_cell_9/recurrent_kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/v/lstm_9/lstm_cell_9/recurrent_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/lstm_9/lstm_cell_9/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/lstm_9/lstm_cell_9/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/m/lstm_10/lstm_cell_10/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/lstm_10/lstm_cell_10/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE,Adam/m/lstm_10/lstm_cell_10/recurrent_kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE,Adam/v/lstm_10/lstm_cell_10/recurrent_kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/lstm_10/lstm_cell_10/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/lstm_10/lstm_cell_10/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_4/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_4/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_4/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_4/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

г0
Г1*

ф	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

░0
▒1*

«	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
й
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp-lstm_9/lstm_cell_9/kernel/Read/ReadVariableOp7lstm_9/lstm_cell_9/recurrent_kernel/Read/ReadVariableOp+lstm_9/lstm_cell_9/bias/Read/ReadVariableOp/lstm_10/lstm_cell_10/kernel/Read/ReadVariableOp9lstm_10/lstm_cell_10/recurrent_kernel/Read/ReadVariableOp-lstm_10/lstm_cell_10/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp4Adam/m/lstm_9/lstm_cell_9/kernel/Read/ReadVariableOp4Adam/v/lstm_9/lstm_cell_9/kernel/Read/ReadVariableOp>Adam/m/lstm_9/lstm_cell_9/recurrent_kernel/Read/ReadVariableOp>Adam/v/lstm_9/lstm_cell_9/recurrent_kernel/Read/ReadVariableOp2Adam/m/lstm_9/lstm_cell_9/bias/Read/ReadVariableOp2Adam/v/lstm_9/lstm_cell_9/bias/Read/ReadVariableOp6Adam/m/lstm_10/lstm_cell_10/kernel/Read/ReadVariableOp6Adam/v/lstm_10/lstm_cell_10/kernel/Read/ReadVariableOp@Adam/m/lstm_10/lstm_cell_10/recurrent_kernel/Read/ReadVariableOp@Adam/v/lstm_10/lstm_cell_10/recurrent_kernel/Read/ReadVariableOp4Adam/m/lstm_10/lstm_cell_10/bias/Read/ReadVariableOp4Adam/v/lstm_10/lstm_cell_10/bias/Read/ReadVariableOp)Adam/m/dense_4/kernel/Read/ReadVariableOp)Adam/v/dense_4/kernel/Read/ReadVariableOp'Adam/m/dense_4/bias/Read/ReadVariableOp'Adam/v/dense_4/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*+
Tin$
"2 	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *(
f#R!
__inference__traced_save_201270
Я
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_4/kerneldense_4/biaslstm_9/lstm_cell_9/kernel#lstm_9/lstm_cell_9/recurrent_kernellstm_9/lstm_cell_9/biaslstm_10/lstm_cell_10/kernel%lstm_10/lstm_cell_10/recurrent_kernellstm_10/lstm_cell_10/bias	iterationlearning_rate Adam/m/lstm_9/lstm_cell_9/kernel Adam/v/lstm_9/lstm_cell_9/kernel*Adam/m/lstm_9/lstm_cell_9/recurrent_kernel*Adam/v/lstm_9/lstm_cell_9/recurrent_kernelAdam/m/lstm_9/lstm_cell_9/biasAdam/v/lstm_9/lstm_cell_9/bias"Adam/m/lstm_10/lstm_cell_10/kernel"Adam/v/lstm_10/lstm_cell_10/kernel,Adam/m/lstm_10/lstm_cell_10/recurrent_kernel,Adam/v/lstm_10/lstm_cell_10/recurrent_kernel Adam/m/lstm_10/lstm_cell_10/bias Adam/v/lstm_10/lstm_cell_10/biasAdam/m/dense_4/kernelAdam/v/dense_4/kernelAdam/m/dense_4/biasAdam/v/dense_4/biastotal_1count_1totalcount**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *+
f&R$
"__inference__traced_restore_201370ЋЄ
▄
Ё
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_201027

inputs
states_0
states_11
matmul_readvariableop_resource:	У3
 matmul_1_readvariableop_resource:	ZУ.
biasadd_readvariableop_resource:	У
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	У*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Уs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         ZN
ReluRelusplit:output:2*
T0*'
_output_shapes
:         Z_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         ZV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         ZK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         Zc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         ZЉ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         Z:         Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         Z
"
_user_specified_name
states_0:QM
'
_output_shapes
:         Z
"
_user_specified_name
states_1
х
├
while_cond_199777
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_199777___redundant_placeholder04
0while_while_cond_199777___redundant_placeholder14
0while_while_cond_199777___redundant_placeholder24
0while_while_cond_199777___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
└
Ћ
(__inference_dense_4_layer_call_fn_200951

inputs
unknown:Z
	unknown_0:
identityѕбStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_198493o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         Z: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
П
є
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_201125

inputs
states_0
states_11
matmul_readvariableop_resource:	ZУ3
 matmul_1_readvariableop_resource:	ZУ.
biasadd_readvariableop_resource:	У
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Уs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         ZN
ReluRelusplit:output:2*
T0*'
_output_shapes
:         Z_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         ZV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         ZK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         Zc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         ZЉ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         Z:         Z:         Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs:QM
'
_output_shapes
:         Z
"
_user_specified_name
states_0:QM
'
_output_shapes
:         Z
"
_user_specified_name
states_1
§8
Ё
C__inference_lstm_10_layer_call_and_return_conditional_losses_198158

inputs&
lstm_cell_10_198074:	ZУ&
lstm_cell_10_198076:	ZУ"
lstm_cell_10_198078:	У
identityѕб$lstm_cell_10/StatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maskш
$lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10_198074lstm_cell_10_198076lstm_cell_10_198078*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         Z:         Z:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_198028n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10_198074lstm_cell_10_198076lstm_cell_10_198078*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_198088*
condR
while_cond_198087*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         Zu
NoOpNoOp%^lstm_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  Z: : : 2L
$lstm_cell_10/StatefulPartitionedCall$lstm_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  Z
 
_user_specified_nameinputs
Н
ё
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_197880

inputs

states
states_11
matmul_readvariableop_resource:	ZУ3
 matmul_1_readvariableop_resource:	ZУ.
biasadd_readvariableop_resource:	У
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Уs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         ZN
ReluRelusplit:output:2*
T0*'
_output_shapes
:         Z_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         ZV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         ZK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         Zc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         ZЉ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         Z:         Z:         Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs:OK
'
_output_shapes
:         Z
 
_user_specified_namestates:OK
'
_output_shapes
:         Z
 
_user_specified_namestates
е9
л
while_body_200540
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	ZУH
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:	ZУC
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	У
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	ZУF
3while_lstm_cell_10_matmul_1_readvariableop_resource:	ZУA
2while_lstm_cell_10_biasadd_readvariableop_resource:	Уѕб)while/lstm_cell_10/BiasAdd/ReadVariableOpб(while/lstm_cell_10/MatMul/ReadVariableOpб*while/lstm_cell_10/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         Z*
element_dtype0Ю
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0║
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УА
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0А
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уъ
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         УЏ
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0Д
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уd
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitz
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Z|
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         Zє
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         Zt
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         Zў
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         ZЇ
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Z|
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         Zq
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Zю
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         Zr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_10/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         Zy
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Zл

while/NoOpNoOp*^while/lstm_cell_10/BiasAdd/ReadVariableOp)^while/lstm_cell_10/MatMul/ReadVariableOp+^while/lstm_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2V
)while/lstm_cell_10/BiasAdd/ReadVariableOp)while/lstm_cell_10/BiasAdd/ReadVariableOp2T
(while/lstm_cell_10/MatMul/ReadVariableOp(while/lstm_cell_10/MatMul/ReadVariableOp2X
*while/lstm_cell_10/MatMul_1/ReadVariableOp*while/lstm_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
е9
л
while_body_200830
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	ZУH
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:	ZУC
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	У
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	ZУF
3while_lstm_cell_10_matmul_1_readvariableop_resource:	ZУA
2while_lstm_cell_10_biasadd_readvariableop_resource:	Уѕб)while/lstm_cell_10/BiasAdd/ReadVariableOpб(while/lstm_cell_10/MatMul/ReadVariableOpб*while/lstm_cell_10/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         Z*
element_dtype0Ю
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0║
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УА
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0А
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уъ
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         УЏ
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0Д
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уd
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitz
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Z|
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         Zє
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         Zt
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         Zў
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         ZЇ
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Z|
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         Zq
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Zю
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         Zr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_10/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         Zy
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Zл

while/NoOpNoOp*^while/lstm_cell_10/BiasAdd/ReadVariableOp)^while/lstm_cell_10/MatMul/ReadVariableOp+^while/lstm_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2V
)while/lstm_cell_10/BiasAdd/ReadVariableOp)while/lstm_cell_10/BiasAdd/ReadVariableOp2T
(while/lstm_cell_10/MatMul/ReadVariableOp(while/lstm_cell_10/MatMul/ReadVariableOp2X
*while/lstm_cell_10/MatMul_1/ReadVariableOp*while/lstm_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
ѓ$
с
while_body_198088
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_10_198112_0:	ZУ.
while_lstm_cell_10_198114_0:	ZУ*
while_lstm_cell_10_198116_0:	У
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_10_198112:	ZУ,
while_lstm_cell_10_198114:	ZУ(
while_lstm_cell_10_198116:	Уѕб*while/lstm_cell_10/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         Z*
element_dtype0│
*while/lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10_198112_0while_lstm_cell_10_198114_0while_lstm_cell_10_198116_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         Z:         Z:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_198028r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_10/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: љ
while/Identity_4Identity3while/lstm_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         Zљ
while/Identity_5Identity3while/lstm_cell_10/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         Zy

while/NoOpNoOp+^while/lstm_cell_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_10_198112while_lstm_cell_10_198112_0"8
while_lstm_cell_10_198114while_lstm_cell_10_198114_0"8
while_lstm_cell_10_198116while_lstm_cell_10_198116_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2X
*while/lstm_cell_10/StatefulPartitionedCall*while/lstm_cell_10/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
х
├
while_cond_198382
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_198382___redundant_placeholder04
0while_while_cond_198382___redundant_placeholder14
0while_while_cond_198382___redundant_placeholder24
0while_while_cond_198382___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
э
х
(__inference_lstm_10_layer_call_fn_200324

inputs
unknown:	ZУ
	unknown_0:	ZУ
	unknown_1:	У
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_10_layer_call_and_return_conditional_losses_198468o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         Z
 
_user_specified_nameinputs
Ж	
╠
-__inference_sequential_4_layer_call_fn_198519
lstm_9_input
unknown:	У
	unknown_0:	ZУ
	unknown_1:	У
	unknown_2:	ZУ
	unknown_3:	ZУ
	unknown_4:	У
	unknown_5:Z
	unknown_6:
identityѕбStatefulPartitionedCall▒
StatefulPartitionedCallStatefulPartitionedCalllstm_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_198500o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_9_input
ЪR
ы
&sequential_4_lstm_10_while_body_197371F
Bsequential_4_lstm_10_while_sequential_4_lstm_10_while_loop_counterL
Hsequential_4_lstm_10_while_sequential_4_lstm_10_while_maximum_iterations*
&sequential_4_lstm_10_while_placeholder,
(sequential_4_lstm_10_while_placeholder_1,
(sequential_4_lstm_10_while_placeholder_2,
(sequential_4_lstm_10_while_placeholder_3E
Asequential_4_lstm_10_while_sequential_4_lstm_10_strided_slice_1_0Ђ
}sequential_4_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_10_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_4_lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0:	ZУ]
Jsequential_4_lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0:	ZУX
Isequential_4_lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0:	У'
#sequential_4_lstm_10_while_identity)
%sequential_4_lstm_10_while_identity_1)
%sequential_4_lstm_10_while_identity_2)
%sequential_4_lstm_10_while_identity_3)
%sequential_4_lstm_10_while_identity_4)
%sequential_4_lstm_10_while_identity_5C
?sequential_4_lstm_10_while_sequential_4_lstm_10_strided_slice_1
{sequential_4_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_10_tensorarrayunstack_tensorlistfromtensorY
Fsequential_4_lstm_10_while_lstm_cell_10_matmul_readvariableop_resource:	ZУ[
Hsequential_4_lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource:	ZУV
Gsequential_4_lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource:	Уѕб>sequential_4/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpб=sequential_4/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpб?sequential_4/lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpЮ
Lsequential_4/lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   Ј
>sequential_4/lstm_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_4_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_10_tensorarrayunstack_tensorlistfromtensor_0&sequential_4_lstm_10_while_placeholderUsequential_4/lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         Z*
element_dtype0К
=sequential_4/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOpHsequential_4_lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0щ
.sequential_4/lstm_10/while/lstm_cell_10/MatMulMatMulEsequential_4/lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_4/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У╦
?sequential_4/lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOpJsequential_4_lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0Я
0sequential_4/lstm_10/while/lstm_cell_10/MatMul_1MatMul(sequential_4_lstm_10_while_placeholder_2Gsequential_4/lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УП
+sequential_4/lstm_10/while/lstm_cell_10/addAddV28sequential_4/lstm_10/while/lstm_cell_10/MatMul:product:0:sequential_4/lstm_10/while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         У┼
>sequential_4/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOpIsequential_4_lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0Т
/sequential_4/lstm_10/while/lstm_cell_10/BiasAddBiasAdd/sequential_4/lstm_10/while/lstm_cell_10/add:z:0Fsequential_4/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уy
7sequential_4/lstm_10/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :«
-sequential_4/lstm_10/while/lstm_cell_10/splitSplit@sequential_4/lstm_10/while/lstm_cell_10/split/split_dim:output:08sequential_4/lstm_10/while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitц
/sequential_4/lstm_10/while/lstm_cell_10/SigmoidSigmoid6sequential_4/lstm_10/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Zд
1sequential_4/lstm_10/while/lstm_cell_10/Sigmoid_1Sigmoid6sequential_4/lstm_10/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         Z┼
+sequential_4/lstm_10/while/lstm_cell_10/mulMul5sequential_4/lstm_10/while/lstm_cell_10/Sigmoid_1:y:0(sequential_4_lstm_10_while_placeholder_3*
T0*'
_output_shapes
:         Zъ
,sequential_4/lstm_10/while/lstm_cell_10/ReluRelu6sequential_4/lstm_10/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         ZО
-sequential_4/lstm_10/while/lstm_cell_10/mul_1Mul3sequential_4/lstm_10/while/lstm_cell_10/Sigmoid:y:0:sequential_4/lstm_10/while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         Z╠
-sequential_4/lstm_10/while/lstm_cell_10/add_1AddV2/sequential_4/lstm_10/while/lstm_cell_10/mul:z:01sequential_4/lstm_10/while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Zд
1sequential_4/lstm_10/while/lstm_cell_10/Sigmoid_2Sigmoid6sequential_4/lstm_10/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         ZЏ
.sequential_4/lstm_10/while/lstm_cell_10/Relu_1Relu1sequential_4/lstm_10/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Z█
-sequential_4/lstm_10/while/lstm_cell_10/mul_2Mul5sequential_4/lstm_10/while/lstm_cell_10/Sigmoid_2:y:0<sequential_4/lstm_10/while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         ZЄ
Esequential_4/lstm_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ┴
?sequential_4/lstm_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_4_lstm_10_while_placeholder_1Nsequential_4/lstm_10/while/TensorArrayV2Write/TensorListSetItem/index:output:01sequential_4/lstm_10/while/lstm_cell_10/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмb
 sequential_4/lstm_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Џ
sequential_4/lstm_10/while/addAddV2&sequential_4_lstm_10_while_placeholder)sequential_4/lstm_10/while/add/y:output:0*
T0*
_output_shapes
: d
"sequential_4/lstm_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :╗
 sequential_4/lstm_10/while/add_1AddV2Bsequential_4_lstm_10_while_sequential_4_lstm_10_while_loop_counter+sequential_4/lstm_10/while/add_1/y:output:0*
T0*
_output_shapes
: ў
#sequential_4/lstm_10/while/IdentityIdentity$sequential_4/lstm_10/while/add_1:z:0 ^sequential_4/lstm_10/while/NoOp*
T0*
_output_shapes
: Й
%sequential_4/lstm_10/while/Identity_1IdentityHsequential_4_lstm_10_while_sequential_4_lstm_10_while_maximum_iterations ^sequential_4/lstm_10/while/NoOp*
T0*
_output_shapes
: ў
%sequential_4/lstm_10/while/Identity_2Identity"sequential_4/lstm_10/while/add:z:0 ^sequential_4/lstm_10/while/NoOp*
T0*
_output_shapes
: ┼
%sequential_4/lstm_10/while/Identity_3IdentityOsequential_4/lstm_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_4/lstm_10/while/NoOp*
T0*
_output_shapes
: И
%sequential_4/lstm_10/while/Identity_4Identity1sequential_4/lstm_10/while/lstm_cell_10/mul_2:z:0 ^sequential_4/lstm_10/while/NoOp*
T0*'
_output_shapes
:         ZИ
%sequential_4/lstm_10/while/Identity_5Identity1sequential_4/lstm_10/while/lstm_cell_10/add_1:z:0 ^sequential_4/lstm_10/while/NoOp*
T0*'
_output_shapes
:         Zц
sequential_4/lstm_10/while/NoOpNoOp?^sequential_4/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp>^sequential_4/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp@^sequential_4/lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#sequential_4_lstm_10_while_identity,sequential_4/lstm_10/while/Identity:output:0"W
%sequential_4_lstm_10_while_identity_1.sequential_4/lstm_10/while/Identity_1:output:0"W
%sequential_4_lstm_10_while_identity_2.sequential_4/lstm_10/while/Identity_2:output:0"W
%sequential_4_lstm_10_while_identity_3.sequential_4/lstm_10/while/Identity_3:output:0"W
%sequential_4_lstm_10_while_identity_4.sequential_4/lstm_10/while/Identity_4:output:0"W
%sequential_4_lstm_10_while_identity_5.sequential_4/lstm_10/while/Identity_5:output:0"ћ
Gsequential_4_lstm_10_while_lstm_cell_10_biasadd_readvariableop_resourceIsequential_4_lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0"ќ
Hsequential_4_lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resourceJsequential_4_lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0"њ
Fsequential_4_lstm_10_while_lstm_cell_10_matmul_readvariableop_resourceHsequential_4_lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0"ё
?sequential_4_lstm_10_while_sequential_4_lstm_10_strided_slice_1Asequential_4_lstm_10_while_sequential_4_lstm_10_strided_slice_1_0"Ч
{sequential_4_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_10_tensorarrayunstack_tensorlistfromtensor}sequential_4_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2ђ
>sequential_4/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp>sequential_4/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp2~
=sequential_4/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp=sequential_4/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp2ѓ
?sequential_4/lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp?sequential_4/lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
╚7
К
while_body_199778
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	УG
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:	ZУB
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	У
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	УE
2while_lstm_cell_9_matmul_1_readvariableop_resource:	ZУ@
1while_lstm_cell_9_biasadd_readvariableop_resource:	Уѕб(while/lstm_cell_9/BiasAdd/ReadVariableOpб'while/lstm_cell_9/MatMul/ReadVariableOpб)while/lstm_cell_9/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Џ
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	У*
dtype0И
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЪ
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0Ъ
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЏ
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         УЎ
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0ц
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уc
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :В
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitx
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Zz
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:         Zё
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         Zr
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:         ZЋ
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         Zі
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Zz
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:         Zo
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         ZЎ
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         Z─
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         Zx
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Z═

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
е9
л
while_body_198383
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	ZУH
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:	ZУC
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	У
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	ZУF
3while_lstm_cell_10_matmul_1_readvariableop_resource:	ZУA
2while_lstm_cell_10_biasadd_readvariableop_resource:	Уѕб)while/lstm_cell_10/BiasAdd/ReadVariableOpб(while/lstm_cell_10/MatMul/ReadVariableOpб*while/lstm_cell_10/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         Z*
element_dtype0Ю
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0║
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УА
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0А
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уъ
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         УЏ
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0Д
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уd
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitz
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Z|
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         Zє
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         Zt
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         Zў
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         ZЇ
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Z|
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         Zq
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Zю
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         Zr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_10/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         Zy
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Zл

while/NoOpNoOp*^while/lstm_cell_10/BiasAdd/ReadVariableOp)^while/lstm_cell_10/MatMul/ReadVariableOp+^while/lstm_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2V
)while/lstm_cell_10/BiasAdd/ReadVariableOp)while/lstm_cell_10/BiasAdd/ReadVariableOp2T
(while/lstm_cell_10/MatMul/ReadVariableOp(while/lstm_cell_10/MatMul/ReadVariableOp2X
*while/lstm_cell_10/MatMul_1/ReadVariableOp*while/lstm_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
х
├
while_cond_200063
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_200063___redundant_placeholder04
0while_while_cond_200063___redundant_placeholder14
0while_while_cond_200063___redundant_placeholder24
0while_while_cond_200063___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
║	
├
$__inference_signature_wrapper_199044
lstm_9_input
unknown:	У
	unknown_0:	ZУ
	unknown_1:	У
	unknown_2:	ZУ
	unknown_3:	ZУ
	unknown_4:	У
	unknown_5:Z
	unknown_6:
identityѕбStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCalllstm_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__wrapped_model_197463o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_9_input
ЕK
Џ
C__inference_lstm_10_layer_call_and_return_conditional_losses_198710

inputs>
+lstm_cell_10_matmul_readvariableop_resource:	ZУ@
-lstm_cell_10_matmul_1_readvariableop_resource:	ZУ;
,lstm_cell_10_biasadd_readvariableop_resource:	У
identityѕб#lstm_cell_10/BiasAdd/ReadVariableOpб"lstm_cell_10/MatMul/ReadVariableOpб$lstm_cell_10/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maskЈ
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0ќ
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЊ
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0љ
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уї
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         УЇ
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0Ћ
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У^
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitn
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Zp
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:         Zw
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         Zh
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:         Zє
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         Z{
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Zp
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:         Ze
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Zі
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_198625*
condR
while_cond_198624*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         Z└
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         Z: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         Z
 
_user_specified_nameinputs
х
├
while_cond_197543
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_197543___redundant_placeholder04
0while_while_cond_197543___redundant_placeholder14
0while_while_cond_197543___redundant_placeholder24
0while_while_cond_197543___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
ЦJ
ќ
B__inference_lstm_9_layer_call_and_return_conditional_losses_199862
inputs_0=
*lstm_cell_9_matmul_readvariableop_resource:	У?
,lstm_cell_9_matmul_1_readvariableop_resource:	ZУ:
+lstm_cell_9_biasadd_readvariableop_resource:	У
identityѕб"lstm_cell_9/BiasAdd/ReadVariableOpб!lstm_cell_9/MatMul/ReadVariableOpб#lstm_cell_9/MatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЇ
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	У*
dtype0ћ
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЉ
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0ј
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЅ
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         УІ
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0њ
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У]
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :┌
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitl
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Zn
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*'
_output_shapes
:         Zu
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         Zf
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*'
_output_shapes
:         ZЃ
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         Zx
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Zn
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*'
_output_shapes
:         Zc
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         ZЄ
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B :  
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_199778*
condR
while_cond_199777*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Zй
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
љO
К
%sequential_4_lstm_9_while_body_197231D
@sequential_4_lstm_9_while_sequential_4_lstm_9_while_loop_counterJ
Fsequential_4_lstm_9_while_sequential_4_lstm_9_while_maximum_iterations)
%sequential_4_lstm_9_while_placeholder+
'sequential_4_lstm_9_while_placeholder_1+
'sequential_4_lstm_9_while_placeholder_2+
'sequential_4_lstm_9_while_placeholder_3C
?sequential_4_lstm_9_while_sequential_4_lstm_9_strided_slice_1_0
{sequential_4_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_9_tensorarrayunstack_tensorlistfromtensor_0Y
Fsequential_4_lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0:	У[
Hsequential_4_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0:	ZУV
Gsequential_4_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0:	У&
"sequential_4_lstm_9_while_identity(
$sequential_4_lstm_9_while_identity_1(
$sequential_4_lstm_9_while_identity_2(
$sequential_4_lstm_9_while_identity_3(
$sequential_4_lstm_9_while_identity_4(
$sequential_4_lstm_9_while_identity_5A
=sequential_4_lstm_9_while_sequential_4_lstm_9_strided_slice_1}
ysequential_4_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_9_tensorarrayunstack_tensorlistfromtensorW
Dsequential_4_lstm_9_while_lstm_cell_9_matmul_readvariableop_resource:	УY
Fsequential_4_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource:	ZУT
Esequential_4_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:	Уѕб<sequential_4/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpб;sequential_4/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpб=sequential_4/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpю
Ksequential_4/lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       і
=sequential_4/lstm_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_4_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_9_tensorarrayunstack_tensorlistfromtensor_0%sequential_4_lstm_9_while_placeholderTsequential_4/lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0├
;sequential_4/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOpFsequential_4_lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	У*
dtype0З
,sequential_4/lstm_9/while/lstm_cell_9/MatMulMatMulDsequential_4/lstm_9/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_4/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УК
=sequential_4/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOpHsequential_4_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0█
.sequential_4/lstm_9/while/lstm_cell_9/MatMul_1MatMul'sequential_4_lstm_9_while_placeholder_2Esequential_4/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УО
)sequential_4/lstm_9/while/lstm_cell_9/addAddV26sequential_4/lstm_9/while/lstm_cell_9/MatMul:product:08sequential_4/lstm_9/while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         У┴
<sequential_4/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOpGsequential_4_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0Я
-sequential_4/lstm_9/while/lstm_cell_9/BiasAddBiasAdd-sequential_4/lstm_9/while/lstm_cell_9/add:z:0Dsequential_4/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уw
5sequential_4/lstm_9/while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :е
+sequential_4/lstm_9/while/lstm_cell_9/splitSplit>sequential_4/lstm_9/while/lstm_cell_9/split/split_dim:output:06sequential_4/lstm_9/while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitа
-sequential_4/lstm_9/while/lstm_cell_9/SigmoidSigmoid4sequential_4/lstm_9/while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Zб
/sequential_4/lstm_9/while/lstm_cell_9/Sigmoid_1Sigmoid4sequential_4/lstm_9/while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:         Z└
)sequential_4/lstm_9/while/lstm_cell_9/mulMul3sequential_4/lstm_9/while/lstm_cell_9/Sigmoid_1:y:0'sequential_4_lstm_9_while_placeholder_3*
T0*'
_output_shapes
:         Zџ
*sequential_4/lstm_9/while/lstm_cell_9/ReluRelu4sequential_4/lstm_9/while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:         ZЛ
+sequential_4/lstm_9/while/lstm_cell_9/mul_1Mul1sequential_4/lstm_9/while/lstm_cell_9/Sigmoid:y:08sequential_4/lstm_9/while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         Zк
+sequential_4/lstm_9/while/lstm_cell_9/add_1AddV2-sequential_4/lstm_9/while/lstm_cell_9/mul:z:0/sequential_4/lstm_9/while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Zб
/sequential_4/lstm_9/while/lstm_cell_9/Sigmoid_2Sigmoid4sequential_4/lstm_9/while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:         ZЌ
,sequential_4/lstm_9/while/lstm_cell_9/Relu_1Relu/sequential_4/lstm_9/while/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         ZН
+sequential_4/lstm_9/while/lstm_cell_9/mul_2Mul3sequential_4/lstm_9/while/lstm_cell_9/Sigmoid_2:y:0:sequential_4/lstm_9/while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         Zћ
>sequential_4/lstm_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_4_lstm_9_while_placeholder_1%sequential_4_lstm_9_while_placeholder/sequential_4/lstm_9/while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмa
sequential_4/lstm_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ў
sequential_4/lstm_9/while/addAddV2%sequential_4_lstm_9_while_placeholder(sequential_4/lstm_9/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_4/lstm_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :и
sequential_4/lstm_9/while/add_1AddV2@sequential_4_lstm_9_while_sequential_4_lstm_9_while_loop_counter*sequential_4/lstm_9/while/add_1/y:output:0*
T0*
_output_shapes
: Ћ
"sequential_4/lstm_9/while/IdentityIdentity#sequential_4/lstm_9/while/add_1:z:0^sequential_4/lstm_9/while/NoOp*
T0*
_output_shapes
: ║
$sequential_4/lstm_9/while/Identity_1IdentityFsequential_4_lstm_9_while_sequential_4_lstm_9_while_maximum_iterations^sequential_4/lstm_9/while/NoOp*
T0*
_output_shapes
: Ћ
$sequential_4/lstm_9/while/Identity_2Identity!sequential_4/lstm_9/while/add:z:0^sequential_4/lstm_9/while/NoOp*
T0*
_output_shapes
: ┬
$sequential_4/lstm_9/while/Identity_3IdentityNsequential_4/lstm_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_4/lstm_9/while/NoOp*
T0*
_output_shapes
: ┤
$sequential_4/lstm_9/while/Identity_4Identity/sequential_4/lstm_9/while/lstm_cell_9/mul_2:z:0^sequential_4/lstm_9/while/NoOp*
T0*'
_output_shapes
:         Z┤
$sequential_4/lstm_9/while/Identity_5Identity/sequential_4/lstm_9/while/lstm_cell_9/add_1:z:0^sequential_4/lstm_9/while/NoOp*
T0*'
_output_shapes
:         ZЮ
sequential_4/lstm_9/while/NoOpNoOp=^sequential_4/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp<^sequential_4/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp>^sequential_4/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"sequential_4_lstm_9_while_identity+sequential_4/lstm_9/while/Identity:output:0"U
$sequential_4_lstm_9_while_identity_1-sequential_4/lstm_9/while/Identity_1:output:0"U
$sequential_4_lstm_9_while_identity_2-sequential_4/lstm_9/while/Identity_2:output:0"U
$sequential_4_lstm_9_while_identity_3-sequential_4/lstm_9/while/Identity_3:output:0"U
$sequential_4_lstm_9_while_identity_4-sequential_4/lstm_9/while/Identity_4:output:0"U
$sequential_4_lstm_9_while_identity_5-sequential_4/lstm_9/while/Identity_5:output:0"љ
Esequential_4_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resourceGsequential_4_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0"њ
Fsequential_4_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resourceHsequential_4_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0"ј
Dsequential_4_lstm_9_while_lstm_cell_9_matmul_readvariableop_resourceFsequential_4_lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0"ђ
=sequential_4_lstm_9_while_sequential_4_lstm_9_strided_slice_1?sequential_4_lstm_9_while_sequential_4_lstm_9_strided_slice_1_0"Э
ysequential_4_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_9_tensorarrayunstack_tensorlistfromtensor{sequential_4_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2|
<sequential_4/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp<sequential_4/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp2z
;sequential_4/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp;sequential_4/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp2~
=sequential_4/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp=sequential_4/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
П
є
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_201157

inputs
states_0
states_11
matmul_readvariableop_resource:	ZУ3
 matmul_1_readvariableop_resource:	ZУ.
biasadd_readvariableop_resource:	У
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Уs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         ZN
ReluRelusplit:output:2*
T0*'
_output_shapes
:         Z_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         ZV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         ZK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         Zc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         ZЉ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         Z:         Z:         Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs:QM
'
_output_shapes
:         Z
"
_user_specified_name
states_0:QM
'
_output_shapes
:         Z
"
_user_specified_name
states_1
х
├
while_cond_200394
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_200394___redundant_placeholder04
0while_while_cond_200394___redundant_placeholder14
0while_while_cond_200394___redundant_placeholder24
0while_while_cond_200394___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
в
Ш
-__inference_lstm_cell_10_layer_call_fn_201076

inputs
states_0
states_1
unknown:	ZУ
	unknown_0:	ZУ
	unknown_1:	У
identity

identity_1

identity_2ѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         Z:         Z:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_197880o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Zq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         Zq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         Z:         Z:         Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs:QM
'
_output_shapes
:         Z
"
_user_specified_name
states_0:QM
'
_output_shapes
:         Z
"
_user_specified_name
states_1
е9
л
while_body_198625
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	ZУH
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:	ZУC
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	У
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	ZУF
3while_lstm_cell_10_matmul_1_readvariableop_resource:	ZУA
2while_lstm_cell_10_biasadd_readvariableop_resource:	Уѕб)while/lstm_cell_10/BiasAdd/ReadVariableOpб(while/lstm_cell_10/MatMul/ReadVariableOpб*while/lstm_cell_10/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         Z*
element_dtype0Ю
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0║
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УА
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0А
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уъ
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         УЏ
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0Д
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уd
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitz
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Z|
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         Zє
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         Zt
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         Zў
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         ZЇ
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Z|
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         Zq
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Zю
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         Zr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_10/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         Zy
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Zл

while/NoOpNoOp*^while/lstm_cell_10/BiasAdd/ReadVariableOp)^while/lstm_cell_10/MatMul/ReadVariableOp+^while/lstm_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2V
)while/lstm_cell_10/BiasAdd/ReadVariableOp)while/lstm_cell_10/BiasAdd/ReadVariableOp2T
(while/lstm_cell_10/MatMul/ReadVariableOp(while/lstm_cell_10/MatMul/ReadVariableOp2X
*while/lstm_cell_10/MatMul_1/ReadVariableOp*while/lstm_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
ЕK
Џ
C__inference_lstm_10_layer_call_and_return_conditional_losses_200915

inputs>
+lstm_cell_10_matmul_readvariableop_resource:	ZУ@
-lstm_cell_10_matmul_1_readvariableop_resource:	ZУ;
,lstm_cell_10_biasadd_readvariableop_resource:	У
identityѕб#lstm_cell_10/BiasAdd/ReadVariableOpб"lstm_cell_10/MatMul/ReadVariableOpб$lstm_cell_10/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maskЈ
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0ќ
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЊ
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0љ
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уї
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         УЇ
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0Ћ
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У^
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitn
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Zp
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:         Zw
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         Zh
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:         Zє
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         Z{
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Zp
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:         Ze
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Zі
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_200830*
condR
while_cond_200829*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         Z└
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         Z: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         Z
 
_user_specified_nameinputs
п
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_200930

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         Z[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         Z"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         Z:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
уI
ћ
B__inference_lstm_9_layer_call_and_return_conditional_losses_200148

inputs=
*lstm_cell_9_matmul_readvariableop_resource:	У?
,lstm_cell_9_matmul_1_readvariableop_resource:	ZУ:
+lstm_cell_9_biasadd_readvariableop_resource:	У
identityѕб"lstm_cell_9/BiasAdd/ReadVariableOpб!lstm_cell_9/MatMul/ReadVariableOpб#lstm_cell_9/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЇ
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	У*
dtype0ћ
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЉ
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0ј
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЅ
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         УІ
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0њ
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У]
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :┌
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitl
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Zn
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*'
_output_shapes
:         Zu
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         Zf
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*'
_output_shapes
:         ZЃ
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         Zx
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Zn
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*'
_output_shapes
:         Zc
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         ZЄ
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B :  
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_200064*
condR
while_cond_200063*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         Zй
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
щ	
¤
lstm_9_while_cond_199435*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3,
(lstm_9_while_less_lstm_9_strided_slice_1B
>lstm_9_while_lstm_9_while_cond_199435___redundant_placeholder0B
>lstm_9_while_lstm_9_while_cond_199435___redundant_placeholder1B
>lstm_9_while_lstm_9_while_cond_199435___redundant_placeholder2B
>lstm_9_while_lstm_9_while_cond_199435___redundant_placeholder3
lstm_9_while_identity
~
lstm_9/while/LessLesslstm_9_while_placeholder(lstm_9_while_less_lstm_9_strided_slice_1*
T0*
_output_shapes
: Y
lstm_9/while/IdentityIdentitylstm_9/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_9_while_identitylstm_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
І

d
E__inference_dropout_4_layer_call_and_return_conditional_losses_198549

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         ZC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         Z*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         ZT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         Za
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:         Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         Z:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
ѓ8
ђ
B__inference_lstm_9_layer_call_and_return_conditional_losses_197804

inputs%
lstm_cell_9_197722:	У%
lstm_cell_9_197724:	ZУ!
lstm_cell_9_197726:	У
identityѕб#lstm_cell_9/StatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask­
#lstm_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_9_197722lstm_cell_9_197724lstm_cell_9_197726*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         Z:         Z:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_197676n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┤
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_9_197722lstm_cell_9_197724lstm_cell_9_197726*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_197735*
condR
while_cond_197734*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Zt
NoOpNoOp$^lstm_cell_9/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_9/StatefulPartitionedCall#lstm_cell_9/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
х
├
while_cond_198087
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_198087___redundant_placeholder04
0while_while_cond_198087___redundant_placeholder14
0while_while_cond_198087___redundant_placeholder24
0while_while_cond_198087___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
╚7
К
while_body_198232
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	УG
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:	ZУB
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	У
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	УE
2while_lstm_cell_9_matmul_1_readvariableop_resource:	ZУ@
1while_lstm_cell_9_biasadd_readvariableop_resource:	Уѕб(while/lstm_cell_9/BiasAdd/ReadVariableOpб'while/lstm_cell_9/MatMul/ReadVariableOpб)while/lstm_cell_9/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Џ
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	У*
dtype0И
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЪ
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0Ъ
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЏ
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         УЎ
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0ц
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уc
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :В
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitx
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Zz
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:         Zё
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         Zr
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:         ZЋ
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         Zі
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Zz
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:         Zo
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         ZЎ
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         Z─
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         Zx
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Z═

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
п	
к
-__inference_sequential_4_layer_call_fn_199065

inputs
unknown:	У
	unknown_0:	ZУ
	unknown_1:	У
	unknown_2:	ZУ
	unknown_3:	ZУ
	unknown_4:	У
	unknown_5:Z
	unknown_6:
identityѕбStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_198500o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Аг
Ь
H__inference_sequential_4_layer_call_and_return_conditional_losses_199377

inputsD
1lstm_9_lstm_cell_9_matmul_readvariableop_resource:	УF
3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource:	ZУA
2lstm_9_lstm_cell_9_biasadd_readvariableop_resource:	УF
3lstm_10_lstm_cell_10_matmul_readvariableop_resource:	ZУH
5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource:	ZУC
4lstm_10_lstm_cell_10_biasadd_readvariableop_resource:	У8
&dense_4_matmul_readvariableop_resource:Z5
'dense_4_biasadd_readvariableop_resource:
identityѕбdense_4/BiasAdd/ReadVariableOpбdense_4/MatMul/ReadVariableOpб+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpб*lstm_10/lstm_cell_10/MatMul/ReadVariableOpб,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpбlstm_10/whileб)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpб(lstm_9/lstm_cell_9/MatMul/ReadVariableOpб*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpбlstm_9/whileB
lstm_9/ShapeShapeinputs*
T0*
_output_shapes
:d
lstm_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
lstm_9/strided_sliceStridedSlicelstm_9/Shape:output:0#lstm_9/strided_slice/stack:output:0%lstm_9/strided_slice/stack_1:output:0%lstm_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zѕ
lstm_9/zeros/packedPacklstm_9/strided_slice:output:0lstm_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ђ
lstm_9/zerosFilllstm_9/zeros/packed:output:0lstm_9/zeros/Const:output:0*
T0*'
_output_shapes
:         ZY
lstm_9/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zї
lstm_9/zeros_1/packedPacklstm_9/strided_slice:output:0 lstm_9/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_9/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Є
lstm_9/zeros_1Filllstm_9/zeros_1/packed:output:0lstm_9/zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zj
lstm_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          {
lstm_9/transpose	Transposeinputslstm_9/transpose/perm:output:0*
T0*+
_output_shapes
:         R
lstm_9/Shape_1Shapelstm_9/transpose:y:0*
T0*
_output_shapes
:f
lstm_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
lstm_9/strided_slice_1StridedSlicelstm_9/Shape_1:output:0%lstm_9/strided_slice_1/stack:output:0'lstm_9/strided_slice_1/stack_1:output:0'lstm_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
lstm_9/TensorArrayV2TensorListReserve+lstm_9/TensorArrayV2/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмЇ
<lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ш
.lstm_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_9/transpose:y:0Elstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмf
lstm_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ї
lstm_9/strided_slice_2StridedSlicelstm_9/transpose:y:0%lstm_9/strided_slice_2/stack:output:0'lstm_9/strided_slice_2/stack_1:output:0'lstm_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЏ
(lstm_9/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp1lstm_9_lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	У*
dtype0Е
lstm_9/lstm_cell_9/MatMulMatMullstm_9/strided_slice_2:output:00lstm_9/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЪ
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0Б
lstm_9/lstm_cell_9/MatMul_1MatMullstm_9/zeros:output:02lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уъ
lstm_9/lstm_cell_9/addAddV2#lstm_9/lstm_cell_9/MatMul:product:0%lstm_9/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         УЎ
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp2lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0Д
lstm_9/lstm_cell_9/BiasAddBiasAddlstm_9/lstm_cell_9/add:z:01lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уd
"lstm_9/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
lstm_9/lstm_cell_9/splitSplit+lstm_9/lstm_cell_9/split/split_dim:output:0#lstm_9/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitz
lstm_9/lstm_cell_9/SigmoidSigmoid!lstm_9/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Z|
lstm_9/lstm_cell_9/Sigmoid_1Sigmoid!lstm_9/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:         Zі
lstm_9/lstm_cell_9/mulMul lstm_9/lstm_cell_9/Sigmoid_1:y:0lstm_9/zeros_1:output:0*
T0*'
_output_shapes
:         Zt
lstm_9/lstm_cell_9/ReluRelu!lstm_9/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:         Zў
lstm_9/lstm_cell_9/mul_1Mullstm_9/lstm_cell_9/Sigmoid:y:0%lstm_9/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         ZЇ
lstm_9/lstm_cell_9/add_1AddV2lstm_9/lstm_cell_9/mul:z:0lstm_9/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Z|
lstm_9/lstm_cell_9/Sigmoid_2Sigmoid!lstm_9/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:         Zq
lstm_9/lstm_cell_9/Relu_1Relulstm_9/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         Zю
lstm_9/lstm_cell_9/mul_2Mul lstm_9/lstm_cell_9/Sigmoid_2:y:0'lstm_9/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         Zu
$lstm_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ═
lstm_9/TensorArrayV2_1TensorListReserve-lstm_9/TensorArrayV2_1/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмM
lstm_9/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         [
lstm_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : р
lstm_9/whileWhile"lstm_9/while/loop_counter:output:0(lstm_9/while/maximum_iterations:output:0lstm_9/time:output:0lstm_9/TensorArrayV2_1:handle:0lstm_9/zeros:output:0lstm_9/zeros_1:output:0lstm_9/strided_slice_1:output:0>lstm_9/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_9_lstm_cell_9_matmul_readvariableop_resource3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource2lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_9_while_body_199145*$
condR
lstm_9_while_cond_199144*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations ѕ
7lstm_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   О
)lstm_9/TensorArrayV2Stack/TensorListStackTensorListStacklstm_9/while:output:3@lstm_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0o
lstm_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         h
lstm_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
lstm_9/strided_slice_3StridedSlice2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_9/strided_slice_3/stack:output:0'lstm_9/strided_slice_3/stack_1:output:0'lstm_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maskl
lstm_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ф
lstm_9/transpose_1	Transpose2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:         Zb
lstm_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    S
lstm_10/ShapeShapelstm_9/transpose_1:y:0*
T0*
_output_shapes
:e
lstm_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_10/strided_sliceStridedSlicelstm_10/Shape:output:0$lstm_10/strided_slice/stack:output:0&lstm_10/strided_slice/stack_1:output:0&lstm_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ZІ
lstm_10/zeros/packedPacklstm_10/strided_slice:output:0lstm_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ё
lstm_10/zerosFilllstm_10/zeros/packed:output:0lstm_10/zeros/Const:output:0*
T0*'
_output_shapes
:         ZZ
lstm_10/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ZЈ
lstm_10/zeros_1/packedPacklstm_10/strided_slice:output:0!lstm_10/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_10/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    і
lstm_10/zeros_1Filllstm_10/zeros_1/packed:output:0lstm_10/zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zk
lstm_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ї
lstm_10/transpose	Transposelstm_9/transpose_1:y:0lstm_10/transpose/perm:output:0*
T0*+
_output_shapes
:         ZT
lstm_10/Shape_1Shapelstm_10/transpose:y:0*
T0*
_output_shapes
:g
lstm_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
lstm_10/strided_slice_1StridedSlicelstm_10/Shape_1:output:0&lstm_10/strided_slice_1/stack:output:0(lstm_10/strided_slice_1/stack_1:output:0(lstm_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_10/TensorArrayV2TensorListReserve,lstm_10/TensorArrayV2/element_shape:output:0 lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмј
=lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   Э
/lstm_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_10/transpose:y:0Flstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмg
lstm_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
lstm_10/strided_slice_2StridedSlicelstm_10/transpose:y:0&lstm_10/strided_slice_2/stack:output:0(lstm_10/strided_slice_2/stack_1:output:0(lstm_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maskЪ
*lstm_10/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3lstm_10_lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0«
lstm_10/lstm_cell_10/MatMulMatMul lstm_10/strided_slice_2:output:02lstm_10/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УБ
,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0е
lstm_10/lstm_cell_10/MatMul_1MatMullstm_10/zeros:output:04lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уц
lstm_10/lstm_cell_10/addAddV2%lstm_10/lstm_cell_10/MatMul:product:0'lstm_10/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         УЮ
+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0Г
lstm_10/lstm_cell_10/BiasAddBiasAddlstm_10/lstm_cell_10/add:z:03lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уf
$lstm_10/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
lstm_10/lstm_cell_10/splitSplit-lstm_10/lstm_cell_10/split/split_dim:output:0%lstm_10/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_split~
lstm_10/lstm_cell_10/SigmoidSigmoid#lstm_10/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Zђ
lstm_10/lstm_cell_10/Sigmoid_1Sigmoid#lstm_10/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         ZЈ
lstm_10/lstm_cell_10/mulMul"lstm_10/lstm_cell_10/Sigmoid_1:y:0lstm_10/zeros_1:output:0*
T0*'
_output_shapes
:         Zx
lstm_10/lstm_cell_10/ReluRelu#lstm_10/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         Zъ
lstm_10/lstm_cell_10/mul_1Mul lstm_10/lstm_cell_10/Sigmoid:y:0'lstm_10/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         ZЊ
lstm_10/lstm_cell_10/add_1AddV2lstm_10/lstm_cell_10/mul:z:0lstm_10/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Zђ
lstm_10/lstm_cell_10/Sigmoid_2Sigmoid#lstm_10/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         Zu
lstm_10/lstm_cell_10/Relu_1Relulstm_10/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Zб
lstm_10/lstm_cell_10/mul_2Mul"lstm_10/lstm_cell_10/Sigmoid_2:y:0)lstm_10/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         Zv
%lstm_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   f
$lstm_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_10/TensorArrayV2_1TensorListReserve.lstm_10/TensorArrayV2_1/element_shape:output:0-lstm_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмN
lstm_10/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ы
lstm_10/whileWhile#lstm_10/while/loop_counter:output:0)lstm_10/while/maximum_iterations:output:0lstm_10/time:output:0 lstm_10/TensorArrayV2_1:handle:0lstm_10/zeros:output:0lstm_10/zeros_1:output:0 lstm_10/strided_slice_1:output:0?lstm_10/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_10_lstm_cell_10_matmul_readvariableop_resource5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource4lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_10_while_body_199285*%
condR
lstm_10_while_cond_199284*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ѕ
8lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   Ь
*lstm_10/TensorArrayV2Stack/TensorListStackTensorListStacklstm_10/while:output:3Alstm_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0*
num_elementsp
lstm_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
lstm_10/strided_slice_3StridedSlice3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_10/strided_slice_3/stack:output:0(lstm_10/strided_slice_3/stack_1:output:0(lstm_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maskm
lstm_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_10/transpose_1	Transpose3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:         Zc
lstm_10/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    r
dropout_4/IdentityIdentity lstm_10/strided_slice_3:output:0*
T0*'
_output_shapes
:         Zё
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype0ј
dense_4/MatMulMatMuldropout_4/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ѓ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ј
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         g
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ┤
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp,^lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp+^lstm_10/lstm_cell_10/MatMul/ReadVariableOp-^lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp^lstm_10/while*^lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp)^lstm_9/lstm_cell_9/MatMul/ReadVariableOp+^lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp^lstm_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2Z
+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp2X
*lstm_10/lstm_cell_10/MatMul/ReadVariableOp*lstm_10/lstm_cell_10/MatMul/ReadVariableOp2\
,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp2
lstm_10/whilelstm_10/while2V
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp2T
(lstm_9/lstm_cell_9/MatMul/ReadVariableOp(lstm_9/lstm_cell_9/MatMul/ReadVariableOp2X
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp2
lstm_9/whilelstm_9/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╚7
К
while_body_200207
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	УG
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:	ZУB
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	У
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	УE
2while_lstm_cell_9_matmul_1_readvariableop_resource:	ZУ@
1while_lstm_cell_9_biasadd_readvariableop_resource:	Уѕб(while/lstm_cell_9/BiasAdd/ReadVariableOpб'while/lstm_cell_9/MatMul/ReadVariableOpб)while/lstm_cell_9/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Џ
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	У*
dtype0И
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЪ
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0Ъ
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЏ
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         УЎ
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0ц
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уc
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :В
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitx
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Zz
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:         Zё
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         Zr
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:         ZЋ
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         Zі
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Zz
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:         Zo
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         ZЎ
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         Z─
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         Zx
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Z═

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
ѓ
у
&sequential_4_lstm_10_while_cond_197370F
Bsequential_4_lstm_10_while_sequential_4_lstm_10_while_loop_counterL
Hsequential_4_lstm_10_while_sequential_4_lstm_10_while_maximum_iterations*
&sequential_4_lstm_10_while_placeholder,
(sequential_4_lstm_10_while_placeholder_1,
(sequential_4_lstm_10_while_placeholder_2,
(sequential_4_lstm_10_while_placeholder_3H
Dsequential_4_lstm_10_while_less_sequential_4_lstm_10_strided_slice_1^
Zsequential_4_lstm_10_while_sequential_4_lstm_10_while_cond_197370___redundant_placeholder0^
Zsequential_4_lstm_10_while_sequential_4_lstm_10_while_cond_197370___redundant_placeholder1^
Zsequential_4_lstm_10_while_sequential_4_lstm_10_while_cond_197370___redundant_placeholder2^
Zsequential_4_lstm_10_while_sequential_4_lstm_10_while_cond_197370___redundant_placeholder3'
#sequential_4_lstm_10_while_identity
Х
sequential_4/lstm_10/while/LessLess&sequential_4_lstm_10_while_placeholderDsequential_4_lstm_10_while_less_sequential_4_lstm_10_strided_slice_1*
T0*
_output_shapes
: u
#sequential_4/lstm_10/while/IdentityIdentity#sequential_4/lstm_10/while/Less:z:0*
T0
*
_output_shapes
: "S
#sequential_4_lstm_10_while_identity,sequential_4/lstm_10/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
ж
ш
,__inference_lstm_cell_9_layer_call_fn_200978

inputs
states_0
states_1
unknown:	У
	unknown_0:	ZУ
	unknown_1:	У
identity

identity_1

identity_2ѕбStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         Z:         Z:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_197530o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Zq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         Zq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         Z:         Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         Z
"
_user_specified_name
states_0:QM
'
_output_shapes
:         Z
"
_user_specified_name
states_1
І

d
E__inference_dropout_4_layer_call_and_return_conditional_losses_200942

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         ZC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         Z*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         ZT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Њ
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         Za
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:         Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         Z:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
ѓ$
с
while_body_197895
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_10_197919_0:	ZУ.
while_lstm_cell_10_197921_0:	ZУ*
while_lstm_cell_10_197923_0:	У
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_10_197919:	ZУ,
while_lstm_cell_10_197921:	ZУ(
while_lstm_cell_10_197923:	Уѕб*while/lstm_cell_10/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         Z*
element_dtype0│
*while/lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10_197919_0while_lstm_cell_10_197921_0while_lstm_cell_10_197923_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         Z:         Z:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_197880r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ё
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:03while/lstm_cell_10/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: љ
while/Identity_4Identity3while/lstm_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         Zљ
while/Identity_5Identity3while/lstm_cell_10/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         Zy

while/NoOpNoOp+^while/lstm_cell_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_10_197919while_lstm_cell_10_197919_0"8
while_lstm_cell_10_197921while_lstm_cell_10_197921_0"8
while_lstm_cell_10_197923while_lstm_cell_10_197923_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2X
*while/lstm_cell_10/StatefulPartitionedCall*while/lstm_cell_10/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
Ј
и
(__inference_lstm_10_layer_call_fn_200313
inputs_0
unknown:	ZУ
	unknown_0:	ZУ
	unknown_1:	У
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_10_layer_call_and_return_conditional_losses_198158o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  Z
"
_user_specified_name
inputs_0
╚7
К
while_body_198791
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	УG
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:	ZУB
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	У
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	УE
2while_lstm_cell_9_matmul_1_readvariableop_resource:	ZУ@
1while_lstm_cell_9_biasadd_readvariableop_resource:	Уѕб(while/lstm_cell_9/BiasAdd/ReadVariableOpб'while/lstm_cell_9/MatMul/ReadVariableOpб)while/lstm_cell_9/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Џ
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	У*
dtype0И
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЪ
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0Ъ
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЏ
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         УЎ
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0ц
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уc
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :В
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitx
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Zz
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:         Zё
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         Zr
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:         ZЋ
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         Zі
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Zz
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:         Zo
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         ZЎ
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         Z─
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         Zx
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Z═

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
к	
З
C__inference_dense_4_layer_call_and_return_conditional_losses_198493

inputs0
matmul_readvariableop_resource:Z-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
ЦJ
ќ
B__inference_lstm_9_layer_call_and_return_conditional_losses_200005
inputs_0=
*lstm_cell_9_matmul_readvariableop_resource:	У?
,lstm_cell_9_matmul_1_readvariableop_resource:	ZУ:
+lstm_cell_9_biasadd_readvariableop_resource:	У
identityѕб"lstm_cell_9/BiasAdd/ReadVariableOpб!lstm_cell_9/MatMul/ReadVariableOpб#lstm_cell_9/MatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЇ
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	У*
dtype0ћ
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЉ
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0ј
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЅ
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         УІ
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0њ
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У]
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :┌
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitl
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Zn
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*'
_output_shapes
:         Zu
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         Zf
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*'
_output_shapes
:         ZЃ
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         Zx
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Zn
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*'
_output_shapes
:         Zc
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         ZЄ
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B :  
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_199921*
condR
while_cond_199920*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Zй
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
е9
л
while_body_200685
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	ZУH
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:	ZУC
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	У
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	ZУF
3while_lstm_cell_10_matmul_1_readvariableop_resource:	ZУA
2while_lstm_cell_10_biasadd_readvariableop_resource:	Уѕб)while/lstm_cell_10/BiasAdd/ReadVariableOpб(while/lstm_cell_10/MatMul/ReadVariableOpб*while/lstm_cell_10/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         Z*
element_dtype0Ю
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0║
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УА
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0А
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уъ
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         УЏ
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0Д
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уd
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitz
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Z|
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         Zє
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         Zt
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         Zў
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         ZЇ
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Z|
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         Zq
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Zю
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         Zr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_10/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         Zy
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Zл

while/NoOpNoOp*^while/lstm_cell_10/BiasAdd/ReadVariableOp)^while/lstm_cell_10/MatMul/ReadVariableOp+^while/lstm_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2V
)while/lstm_cell_10/BiasAdd/ReadVariableOp)while/lstm_cell_10/BiasAdd/ReadVariableOp2T
(while/lstm_cell_10/MatMul/ReadVariableOp(while/lstm_cell_10/MatMul/ReadVariableOp2X
*while/lstm_cell_10/MatMul_1/ReadVariableOp*while/lstm_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
ќ

с
lstm_10_while_cond_199284,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3.
*lstm_10_while_less_lstm_10_strided_slice_1D
@lstm_10_while_lstm_10_while_cond_199284___redundant_placeholder0D
@lstm_10_while_lstm_10_while_cond_199284___redundant_placeholder1D
@lstm_10_while_lstm_10_while_cond_199284___redundant_placeholder2D
@lstm_10_while_lstm_10_while_cond_199284___redundant_placeholder3
lstm_10_while_identity
ѓ
lstm_10/while/LessLesslstm_10_while_placeholder*lstm_10_while_less_lstm_10_strided_slice_1*
T0*
_output_shapes
: [
lstm_10/while/IdentityIdentitylstm_10/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_10_while_identitylstm_10/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
у
§
H__inference_sequential_4_layer_call_and_return_conditional_losses_199019
lstm_9_input 
lstm_9_198998:	У 
lstm_9_199000:	ZУ
lstm_9_199002:	У!
lstm_10_199005:	ZУ!
lstm_10_199007:	ZУ
lstm_10_199009:	У 
dense_4_199013:Z
dense_4_199015:
identityѕбdense_4/StatefulPartitionedCallб!dropout_4/StatefulPartitionedCallбlstm_10/StatefulPartitionedCallбlstm_9/StatefulPartitionedCallЃ
lstm_9/StatefulPartitionedCallStatefulPartitionedCalllstm_9_inputlstm_9_198998lstm_9_199000lstm_9_199002*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_198875Ъ
lstm_10/StatefulPartitionedCallStatefulPartitionedCall'lstm_9/StatefulPartitionedCall:output:0lstm_10_199005lstm_10_199007lstm_10_199009*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_10_layer_call_and_return_conditional_losses_198710В
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(lstm_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_198549љ
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_4_199013dense_4_199015*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_198493w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ¤
NoOpNoOp ^dense_4/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_9_input
Н
ё
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_198028

inputs

states
states_11
matmul_readvariableop_resource:	ZУ3
 matmul_1_readvariableop_resource:	ZУ.
biasadd_readvariableop_resource:	У
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Уs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         ZN
ReluRelusplit:output:2*
T0*'
_output_shapes
:         Z_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         ZV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         ZK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         Zc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         ZЉ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         Z:         Z:         Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs:OK
'
_output_shapes
:         Z
 
_user_specified_namestates:OK
'
_output_shapes
:         Z
 
_user_specified_namestates
Н
э
H__inference_sequential_4_layer_call_and_return_conditional_losses_198931

inputs 
lstm_9_198910:	У 
lstm_9_198912:	ZУ
lstm_9_198914:	У!
lstm_10_198917:	ZУ!
lstm_10_198919:	ZУ
lstm_10_198921:	У 
dense_4_198925:Z
dense_4_198927:
identityѕбdense_4/StatefulPartitionedCallб!dropout_4/StatefulPartitionedCallбlstm_10/StatefulPartitionedCallбlstm_9/StatefulPartitionedCall§
lstm_9/StatefulPartitionedCallStatefulPartitionedCallinputslstm_9_198910lstm_9_198912lstm_9_198914*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_198875Ъ
lstm_10/StatefulPartitionedCallStatefulPartitionedCall'lstm_9/StatefulPartitionedCall:output:0lstm_10_198917lstm_10_198919lstm_10_198921*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_10_layer_call_and_return_conditional_losses_198710В
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(lstm_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_198549љ
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_4_198925dense_4_198927*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_198493w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ¤
NoOpNoOp ^dense_4/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
▄
Ё
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_201059

inputs
states_0
states_11
matmul_readvariableop_resource:	У3
 matmul_1_readvariableop_resource:	ZУ.
biasadd_readvariableop_resource:	У
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	У*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Уs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         ZN
ReluRelusplit:output:2*
T0*'
_output_shapes
:         Z_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         ZV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         ZK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         Zc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         ZЉ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         Z:         Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         Z
"
_user_specified_name
states_0:QM
'
_output_shapes
:         Z
"
_user_specified_name
states_1
§
┤
'__inference_lstm_9_layer_call_fn_199708

inputs
unknown:	У
	unknown_0:	ZУ
	unknown_1:	У
identityѕбStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_198316s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
п
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_198481

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         Z[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         Z"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         Z:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
н
Ѓ
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_197676

inputs

states
states_11
matmul_readvariableop_resource:	У3
 matmul_1_readvariableop_resource:	ZУ.
biasadd_readvariableop_resource:	У
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	У*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Уs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         ZN
ReluRelusplit:output:2*
T0*'
_output_shapes
:         Z_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         ZV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         ZK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         Zc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         ZЉ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         Z:         Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         Z
 
_user_specified_namestates:OK
'
_output_shapes
:         Z
 
_user_specified_namestates
╚7
К
while_body_199921
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	УG
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:	ZУB
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	У
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	УE
2while_lstm_cell_9_matmul_1_readvariableop_resource:	ZУ@
1while_lstm_cell_9_biasadd_readvariableop_resource:	Уѕб(while/lstm_cell_9/BiasAdd/ReadVariableOpб'while/lstm_cell_9/MatMul/ReadVariableOpб)while/lstm_cell_9/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Џ
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	У*
dtype0И
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЪ
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0Ъ
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЏ
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         УЎ
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0ц
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уc
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :В
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitx
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Zz
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:         Zё
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         Zr
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:         ZЋ
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         Zі
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Zz
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:         Zo
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         ZЎ
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         Z─
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         Zx
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Z═

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
ќ

с
lstm_10_while_cond_199575,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3.
*lstm_10_while_less_lstm_10_strided_slice_1D
@lstm_10_while_lstm_10_while_cond_199575___redundant_placeholder0D
@lstm_10_while_lstm_10_while_cond_199575___redundant_placeholder1D
@lstm_10_while_lstm_10_while_cond_199575___redundant_placeholder2D
@lstm_10_while_lstm_10_while_cond_199575___redundant_placeholder3
lstm_10_while_identity
ѓ
lstm_10/while/LessLesslstm_10_while_placeholder*lstm_10_while_less_lstm_10_strided_slice_1*
T0*
_output_shapes
: [
lstm_10/while/IdentityIdentitylstm_10/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_10_while_identitylstm_10/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
§
┤
'__inference_lstm_9_layer_call_fn_199719

inputs
unknown:	У
	unknown_0:	ZУ
	unknown_1:	У
identityѕбStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_198875s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Ж?
Д

lstm_9_while_body_199436*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3)
%lstm_9_while_lstm_9_strided_slice_1_0e
alstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0:	УN
;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0:	ZУI
:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0:	У
lstm_9_while_identity
lstm_9_while_identity_1
lstm_9_while_identity_2
lstm_9_while_identity_3
lstm_9_while_identity_4
lstm_9_while_identity_5'
#lstm_9_while_lstm_9_strided_slice_1c
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensorJ
7lstm_9_while_lstm_cell_9_matmul_readvariableop_resource:	УL
9lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource:	ZУG
8lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:	Уѕб/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpб.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpб0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpЈ
>lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╔
0lstm_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0lstm_9_while_placeholderGlstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Е
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	У*
dtype0═
lstm_9/while/lstm_cell_9/MatMulMatMul7lstm_9/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УГ
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0┤
!lstm_9/while/lstm_cell_9/MatMul_1MatMullstm_9_while_placeholder_28lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У░
lstm_9/while/lstm_cell_9/addAddV2)lstm_9/while/lstm_cell_9/MatMul:product:0+lstm_9/while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         УД
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0╣
 lstm_9/while/lstm_cell_9/BiasAddBiasAdd lstm_9/while/lstm_cell_9/add:z:07lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уj
(lstm_9/while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ђ
lstm_9/while/lstm_cell_9/splitSplit1lstm_9/while/lstm_cell_9/split/split_dim:output:0)lstm_9/while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitє
 lstm_9/while/lstm_cell_9/SigmoidSigmoid'lstm_9/while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Zѕ
"lstm_9/while/lstm_cell_9/Sigmoid_1Sigmoid'lstm_9/while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:         ZЎ
lstm_9/while/lstm_cell_9/mulMul&lstm_9/while/lstm_cell_9/Sigmoid_1:y:0lstm_9_while_placeholder_3*
T0*'
_output_shapes
:         Zђ
lstm_9/while/lstm_cell_9/ReluRelu'lstm_9/while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:         Zф
lstm_9/while/lstm_cell_9/mul_1Mul$lstm_9/while/lstm_cell_9/Sigmoid:y:0+lstm_9/while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         ZЪ
lstm_9/while/lstm_cell_9/add_1AddV2 lstm_9/while/lstm_cell_9/mul:z:0"lstm_9/while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Zѕ
"lstm_9/while/lstm_cell_9/Sigmoid_2Sigmoid'lstm_9/while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:         Z}
lstm_9/while/lstm_cell_9/Relu_1Relu"lstm_9/while/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         Z«
lstm_9/while/lstm_cell_9/mul_2Mul&lstm_9/while/lstm_cell_9/Sigmoid_2:y:0-lstm_9/while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         ZЯ
1lstm_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_9_while_placeholder_1lstm_9_while_placeholder"lstm_9/while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмT
lstm_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_9/while/addAddV2lstm_9_while_placeholderlstm_9/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
lstm_9/while/add_1AddV2&lstm_9_while_lstm_9_while_loop_counterlstm_9/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_9/while/IdentityIdentitylstm_9/while/add_1:z:0^lstm_9/while/NoOp*
T0*
_output_shapes
: є
lstm_9/while/Identity_1Identity,lstm_9_while_lstm_9_while_maximum_iterations^lstm_9/while/NoOp*
T0*
_output_shapes
: n
lstm_9/while/Identity_2Identitylstm_9/while/add:z:0^lstm_9/while/NoOp*
T0*
_output_shapes
: Џ
lstm_9/while/Identity_3IdentityAlstm_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_9/while/NoOp*
T0*
_output_shapes
: Ї
lstm_9/while/Identity_4Identity"lstm_9/while/lstm_cell_9/mul_2:z:0^lstm_9/while/NoOp*
T0*'
_output_shapes
:         ZЇ
lstm_9/while/Identity_5Identity"lstm_9/while/lstm_cell_9/add_1:z:0^lstm_9/while/NoOp*
T0*'
_output_shapes
:         Zж
lstm_9/while/NoOpNoOp0^lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp/^lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp1^lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_9_while_identitylstm_9/while/Identity:output:0";
lstm_9_while_identity_1 lstm_9/while/Identity_1:output:0";
lstm_9_while_identity_2 lstm_9/while/Identity_2:output:0";
lstm_9_while_identity_3 lstm_9/while/Identity_3:output:0";
lstm_9_while_identity_4 lstm_9/while/Identity_4:output:0";
lstm_9_while_identity_5 lstm_9/while/Identity_5:output:0"L
#lstm_9_while_lstm_9_strided_slice_1%lstm_9_while_lstm_9_strided_slice_1_0"v
8lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0"x
9lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0"t
7lstm_9_while_lstm_cell_9_matmul_readvariableop_resource9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0"─
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensoralstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2b
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp2`
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp2d
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
Ж?
Д

lstm_9_while_body_199145*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3)
%lstm_9_while_lstm_9_strided_slice_1_0e
alstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0:	УN
;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0:	ZУI
:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0:	У
lstm_9_while_identity
lstm_9_while_identity_1
lstm_9_while_identity_2
lstm_9_while_identity_3
lstm_9_while_identity_4
lstm_9_while_identity_5'
#lstm_9_while_lstm_9_strided_slice_1c
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensorJ
7lstm_9_while_lstm_cell_9_matmul_readvariableop_resource:	УL
9lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource:	ZУG
8lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:	Уѕб/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpб.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpб0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpЈ
>lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╔
0lstm_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0lstm_9_while_placeholderGlstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Е
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	У*
dtype0═
lstm_9/while/lstm_cell_9/MatMulMatMul7lstm_9/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УГ
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0┤
!lstm_9/while/lstm_cell_9/MatMul_1MatMullstm_9_while_placeholder_28lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У░
lstm_9/while/lstm_cell_9/addAddV2)lstm_9/while/lstm_cell_9/MatMul:product:0+lstm_9/while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         УД
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0╣
 lstm_9/while/lstm_cell_9/BiasAddBiasAdd lstm_9/while/lstm_cell_9/add:z:07lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уj
(lstm_9/while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ђ
lstm_9/while/lstm_cell_9/splitSplit1lstm_9/while/lstm_cell_9/split/split_dim:output:0)lstm_9/while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitє
 lstm_9/while/lstm_cell_9/SigmoidSigmoid'lstm_9/while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Zѕ
"lstm_9/while/lstm_cell_9/Sigmoid_1Sigmoid'lstm_9/while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:         ZЎ
lstm_9/while/lstm_cell_9/mulMul&lstm_9/while/lstm_cell_9/Sigmoid_1:y:0lstm_9_while_placeholder_3*
T0*'
_output_shapes
:         Zђ
lstm_9/while/lstm_cell_9/ReluRelu'lstm_9/while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:         Zф
lstm_9/while/lstm_cell_9/mul_1Mul$lstm_9/while/lstm_cell_9/Sigmoid:y:0+lstm_9/while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         ZЪ
lstm_9/while/lstm_cell_9/add_1AddV2 lstm_9/while/lstm_cell_9/mul:z:0"lstm_9/while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Zѕ
"lstm_9/while/lstm_cell_9/Sigmoid_2Sigmoid'lstm_9/while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:         Z}
lstm_9/while/lstm_cell_9/Relu_1Relu"lstm_9/while/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         Z«
lstm_9/while/lstm_cell_9/mul_2Mul&lstm_9/while/lstm_cell_9/Sigmoid_2:y:0-lstm_9/while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         ZЯ
1lstm_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_9_while_placeholder_1lstm_9_while_placeholder"lstm_9/while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмT
lstm_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_9/while/addAddV2lstm_9_while_placeholderlstm_9/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
lstm_9/while/add_1AddV2&lstm_9_while_lstm_9_while_loop_counterlstm_9/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_9/while/IdentityIdentitylstm_9/while/add_1:z:0^lstm_9/while/NoOp*
T0*
_output_shapes
: є
lstm_9/while/Identity_1Identity,lstm_9_while_lstm_9_while_maximum_iterations^lstm_9/while/NoOp*
T0*
_output_shapes
: n
lstm_9/while/Identity_2Identitylstm_9/while/add:z:0^lstm_9/while/NoOp*
T0*
_output_shapes
: Џ
lstm_9/while/Identity_3IdentityAlstm_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_9/while/NoOp*
T0*
_output_shapes
: Ї
lstm_9/while/Identity_4Identity"lstm_9/while/lstm_cell_9/mul_2:z:0^lstm_9/while/NoOp*
T0*'
_output_shapes
:         ZЇ
lstm_9/while/Identity_5Identity"lstm_9/while/lstm_cell_9/add_1:z:0^lstm_9/while/NoOp*
T0*'
_output_shapes
:         Zж
lstm_9/while/NoOpNoOp0^lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp/^lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp1^lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_9_while_identitylstm_9/while/Identity:output:0";
lstm_9_while_identity_1 lstm_9/while/Identity_1:output:0";
lstm_9_while_identity_2 lstm_9/while/Identity_2:output:0";
lstm_9_while_identity_3 lstm_9/while/Identity_3:output:0";
lstm_9_while_identity_4 lstm_9/while/Identity_4:output:0";
lstm_9_while_identity_5 lstm_9/while/Identity_5:output:0"L
#lstm_9_while_lstm_9_strided_slice_1%lstm_9_while_lstm_9_strided_slice_1_0"v
8lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0"x
9lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0"t
7lstm_9_while_lstm_cell_9_matmul_readvariableop_resource9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0"─
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensoralstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2b
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp2`
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp2d
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
Ж	
╠
-__inference_sequential_4_layer_call_fn_198971
lstm_9_input
unknown:	У
	unknown_0:	ZУ
	unknown_1:	У
	unknown_2:	ZУ
	unknown_3:	ZУ
	unknown_4:	У
	unknown_5:Z
	unknown_6:
identityѕбStatefulPartitionedCall▒
StatefulPartitionedCallStatefulPartitionedCalllstm_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_198931o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_9_input
уI
ћ
B__inference_lstm_9_layer_call_and_return_conditional_losses_200291

inputs=
*lstm_cell_9_matmul_readvariableop_resource:	У?
,lstm_cell_9_matmul_1_readvariableop_resource:	ZУ:
+lstm_cell_9_biasadd_readvariableop_resource:	У
identityѕб"lstm_cell_9/BiasAdd/ReadVariableOpб!lstm_cell_9/MatMul/ReadVariableOpб#lstm_cell_9/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЇ
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	У*
dtype0ћ
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЉ
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0ј
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЅ
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         УІ
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0њ
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У]
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :┌
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitl
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Zn
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*'
_output_shapes
:         Zu
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         Zf
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*'
_output_shapes
:         ZЃ
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         Zx
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Zn
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*'
_output_shapes
:         Zc
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         ZЄ
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B :  
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_200207*
condR
while_cond_200206*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         Zй
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
«╬
и	
!__inference__wrapped_model_197463
lstm_9_inputQ
>sequential_4_lstm_9_lstm_cell_9_matmul_readvariableop_resource:	УS
@sequential_4_lstm_9_lstm_cell_9_matmul_1_readvariableop_resource:	ZУN
?sequential_4_lstm_9_lstm_cell_9_biasadd_readvariableop_resource:	УS
@sequential_4_lstm_10_lstm_cell_10_matmul_readvariableop_resource:	ZУU
Bsequential_4_lstm_10_lstm_cell_10_matmul_1_readvariableop_resource:	ZУP
Asequential_4_lstm_10_lstm_cell_10_biasadd_readvariableop_resource:	УE
3sequential_4_dense_4_matmul_readvariableop_resource:ZB
4sequential_4_dense_4_biasadd_readvariableop_resource:
identityѕб+sequential_4/dense_4/BiasAdd/ReadVariableOpб*sequential_4/dense_4/MatMul/ReadVariableOpб8sequential_4/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpб7sequential_4/lstm_10/lstm_cell_10/MatMul/ReadVariableOpб9sequential_4/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpбsequential_4/lstm_10/whileб6sequential_4/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpб5sequential_4/lstm_9/lstm_cell_9/MatMul/ReadVariableOpб7sequential_4/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpбsequential_4/lstm_9/whileU
sequential_4/lstm_9/ShapeShapelstm_9_input*
T0*
_output_shapes
:q
'sequential_4/lstm_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_4/lstm_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_4/lstm_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
!sequential_4/lstm_9/strided_sliceStridedSlice"sequential_4/lstm_9/Shape:output:00sequential_4/lstm_9/strided_slice/stack:output:02sequential_4/lstm_9/strided_slice/stack_1:output:02sequential_4/lstm_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_4/lstm_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z»
 sequential_4/lstm_9/zeros/packedPack*sequential_4/lstm_9/strided_slice:output:0+sequential_4/lstm_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_4/lstm_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    е
sequential_4/lstm_9/zerosFill)sequential_4/lstm_9/zeros/packed:output:0(sequential_4/lstm_9/zeros/Const:output:0*
T0*'
_output_shapes
:         Zf
$sequential_4/lstm_9/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z│
"sequential_4/lstm_9/zeros_1/packedPack*sequential_4/lstm_9/strided_slice:output:0-sequential_4/lstm_9/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_4/lstm_9/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
sequential_4/lstm_9/zeros_1Fill+sequential_4/lstm_9/zeros_1/packed:output:0*sequential_4/lstm_9/zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zw
"sequential_4/lstm_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Џ
sequential_4/lstm_9/transpose	Transposelstm_9_input+sequential_4/lstm_9/transpose/perm:output:0*
T0*+
_output_shapes
:         l
sequential_4/lstm_9/Shape_1Shape!sequential_4/lstm_9/transpose:y:0*
T0*
_output_shapes
:s
)sequential_4/lstm_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/lstm_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_4/lstm_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#sequential_4/lstm_9/strided_slice_1StridedSlice$sequential_4/lstm_9/Shape_1:output:02sequential_4/lstm_9/strided_slice_1/stack:output:04sequential_4/lstm_9/strided_slice_1/stack_1:output:04sequential_4/lstm_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_4/lstm_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ­
!sequential_4/lstm_9/TensorArrayV2TensorListReserve8sequential_4/lstm_9/TensorArrayV2/element_shape:output:0,sequential_4/lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмџ
Isequential_4/lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ю
;sequential_4/lstm_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_4/lstm_9/transpose:y:0Rsequential_4/lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмs
)sequential_4/lstm_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/lstm_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_4/lstm_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:═
#sequential_4/lstm_9/strided_slice_2StridedSlice!sequential_4/lstm_9/transpose:y:02sequential_4/lstm_9/strided_slice_2/stack:output:04sequential_4/lstm_9/strided_slice_2/stack_1:output:04sequential_4/lstm_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskх
5sequential_4/lstm_9/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp>sequential_4_lstm_9_lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	У*
dtype0л
&sequential_4/lstm_9/lstm_cell_9/MatMulMatMul,sequential_4/lstm_9/strided_slice_2:output:0=sequential_4/lstm_9/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У╣
7sequential_4/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp@sequential_4_lstm_9_lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0╩
(sequential_4/lstm_9/lstm_cell_9/MatMul_1MatMul"sequential_4/lstm_9/zeros:output:0?sequential_4/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У┼
#sequential_4/lstm_9/lstm_cell_9/addAddV20sequential_4/lstm_9/lstm_cell_9/MatMul:product:02sequential_4/lstm_9/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         У│
6sequential_4/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp?sequential_4_lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0╬
'sequential_4/lstm_9/lstm_cell_9/BiasAddBiasAdd'sequential_4/lstm_9/lstm_cell_9/add:z:0>sequential_4/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уq
/sequential_4/lstm_9/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ќ
%sequential_4/lstm_9/lstm_cell_9/splitSplit8sequential_4/lstm_9/lstm_cell_9/split/split_dim:output:00sequential_4/lstm_9/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitћ
'sequential_4/lstm_9/lstm_cell_9/SigmoidSigmoid.sequential_4/lstm_9/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Zќ
)sequential_4/lstm_9/lstm_cell_9/Sigmoid_1Sigmoid.sequential_4/lstm_9/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:         Z▒
#sequential_4/lstm_9/lstm_cell_9/mulMul-sequential_4/lstm_9/lstm_cell_9/Sigmoid_1:y:0$sequential_4/lstm_9/zeros_1:output:0*
T0*'
_output_shapes
:         Zј
$sequential_4/lstm_9/lstm_cell_9/ReluRelu.sequential_4/lstm_9/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:         Z┐
%sequential_4/lstm_9/lstm_cell_9/mul_1Mul+sequential_4/lstm_9/lstm_cell_9/Sigmoid:y:02sequential_4/lstm_9/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         Z┤
%sequential_4/lstm_9/lstm_cell_9/add_1AddV2'sequential_4/lstm_9/lstm_cell_9/mul:z:0)sequential_4/lstm_9/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Zќ
)sequential_4/lstm_9/lstm_cell_9/Sigmoid_2Sigmoid.sequential_4/lstm_9/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:         ZІ
&sequential_4/lstm_9/lstm_cell_9/Relu_1Relu)sequential_4/lstm_9/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         Z├
%sequential_4/lstm_9/lstm_cell_9/mul_2Mul-sequential_4/lstm_9/lstm_cell_9/Sigmoid_2:y:04sequential_4/lstm_9/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         Zѓ
1sequential_4/lstm_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   З
#sequential_4/lstm_9/TensorArrayV2_1TensorListReserve:sequential_4/lstm_9/TensorArrayV2_1/element_shape:output:0,sequential_4/lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмZ
sequential_4/lstm_9/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_4/lstm_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         h
&sequential_4/lstm_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ќ
sequential_4/lstm_9/whileWhile/sequential_4/lstm_9/while/loop_counter:output:05sequential_4/lstm_9/while/maximum_iterations:output:0!sequential_4/lstm_9/time:output:0,sequential_4/lstm_9/TensorArrayV2_1:handle:0"sequential_4/lstm_9/zeros:output:0$sequential_4/lstm_9/zeros_1:output:0,sequential_4/lstm_9/strided_slice_1:output:0Ksequential_4/lstm_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_4_lstm_9_lstm_cell_9_matmul_readvariableop_resource@sequential_4_lstm_9_lstm_cell_9_matmul_1_readvariableop_resource?sequential_4_lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *1
body)R'
%sequential_4_lstm_9_while_body_197231*1
cond)R'
%sequential_4_lstm_9_while_cond_197230*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ћ
Dsequential_4/lstm_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ■
6sequential_4/lstm_9/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_4/lstm_9/while:output:3Msequential_4/lstm_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0|
)sequential_4/lstm_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         u
+sequential_4/lstm_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/lstm_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
#sequential_4/lstm_9/strided_slice_3StridedSlice?sequential_4/lstm_9/TensorArrayV2Stack/TensorListStack:tensor:02sequential_4/lstm_9/strided_slice_3/stack:output:04sequential_4/lstm_9/strided_slice_3/stack_1:output:04sequential_4/lstm_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_masky
$sequential_4/lstm_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          м
sequential_4/lstm_9/transpose_1	Transpose?sequential_4/lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_4/lstm_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:         Zo
sequential_4/lstm_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    m
sequential_4/lstm_10/ShapeShape#sequential_4/lstm_9/transpose_1:y:0*
T0*
_output_shapes
:r
(sequential_4/lstm_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_4/lstm_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_4/lstm_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"sequential_4/lstm_10/strided_sliceStridedSlice#sequential_4/lstm_10/Shape:output:01sequential_4/lstm_10/strided_slice/stack:output:03sequential_4/lstm_10/strided_slice/stack_1:output:03sequential_4/lstm_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sequential_4/lstm_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Z▓
!sequential_4/lstm_10/zeros/packedPack+sequential_4/lstm_10/strided_slice:output:0,sequential_4/lstm_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 sequential_4/lstm_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ф
sequential_4/lstm_10/zerosFill*sequential_4/lstm_10/zeros/packed:output:0)sequential_4/lstm_10/zeros/Const:output:0*
T0*'
_output_shapes
:         Zg
%sequential_4/lstm_10/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ZХ
#sequential_4/lstm_10/zeros_1/packedPack+sequential_4/lstm_10/strided_slice:output:0.sequential_4/lstm_10/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:g
"sequential_4/lstm_10/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ▒
sequential_4/lstm_10/zeros_1Fill,sequential_4/lstm_10/zeros_1/packed:output:0+sequential_4/lstm_10/zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zx
#sequential_4/lstm_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ┤
sequential_4/lstm_10/transpose	Transpose#sequential_4/lstm_9/transpose_1:y:0,sequential_4/lstm_10/transpose/perm:output:0*
T0*+
_output_shapes
:         Zn
sequential_4/lstm_10/Shape_1Shape"sequential_4/lstm_10/transpose:y:0*
T0*
_output_shapes
:t
*sequential_4/lstm_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_4/lstm_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_4/lstm_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:─
$sequential_4/lstm_10/strided_slice_1StridedSlice%sequential_4/lstm_10/Shape_1:output:03sequential_4/lstm_10/strided_slice_1/stack:output:05sequential_4/lstm_10/strided_slice_1/stack_1:output:05sequential_4/lstm_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0sequential_4/lstm_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         з
"sequential_4/lstm_10/TensorArrayV2TensorListReserve9sequential_4/lstm_10/TensorArrayV2/element_shape:output:0-sequential_4/lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмЏ
Jsequential_4/lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   Ъ
<sequential_4/lstm_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_4/lstm_10/transpose:y:0Ssequential_4/lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмt
*sequential_4/lstm_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_4/lstm_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_4/lstm_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
$sequential_4/lstm_10/strided_slice_2StridedSlice"sequential_4/lstm_10/transpose:y:03sequential_4/lstm_10/strided_slice_2/stack:output:05sequential_4/lstm_10/strided_slice_2/stack_1:output:05sequential_4/lstm_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_mask╣
7sequential_4/lstm_10/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp@sequential_4_lstm_10_lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0Н
(sequential_4/lstm_10/lstm_cell_10/MatMulMatMul-sequential_4/lstm_10/strided_slice_2:output:0?sequential_4/lstm_10/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уй
9sequential_4/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOpBsequential_4_lstm_10_lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0¤
*sequential_4/lstm_10/lstm_cell_10/MatMul_1MatMul#sequential_4/lstm_10/zeros:output:0Asequential_4/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У╦
%sequential_4/lstm_10/lstm_cell_10/addAddV22sequential_4/lstm_10/lstm_cell_10/MatMul:product:04sequential_4/lstm_10/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Уи
8sequential_4/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOpAsequential_4_lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0н
)sequential_4/lstm_10/lstm_cell_10/BiasAddBiasAdd)sequential_4/lstm_10/lstm_cell_10/add:z:0@sequential_4/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уs
1sequential_4/lstm_10/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ю
'sequential_4/lstm_10/lstm_cell_10/splitSplit:sequential_4/lstm_10/lstm_cell_10/split/split_dim:output:02sequential_4/lstm_10/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitў
)sequential_4/lstm_10/lstm_cell_10/SigmoidSigmoid0sequential_4/lstm_10/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Zџ
+sequential_4/lstm_10/lstm_cell_10/Sigmoid_1Sigmoid0sequential_4/lstm_10/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         ZХ
%sequential_4/lstm_10/lstm_cell_10/mulMul/sequential_4/lstm_10/lstm_cell_10/Sigmoid_1:y:0%sequential_4/lstm_10/zeros_1:output:0*
T0*'
_output_shapes
:         Zњ
&sequential_4/lstm_10/lstm_cell_10/ReluRelu0sequential_4/lstm_10/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         Z┼
'sequential_4/lstm_10/lstm_cell_10/mul_1Mul-sequential_4/lstm_10/lstm_cell_10/Sigmoid:y:04sequential_4/lstm_10/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         Z║
'sequential_4/lstm_10/lstm_cell_10/add_1AddV2)sequential_4/lstm_10/lstm_cell_10/mul:z:0+sequential_4/lstm_10/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Zџ
+sequential_4/lstm_10/lstm_cell_10/Sigmoid_2Sigmoid0sequential_4/lstm_10/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         ZЈ
(sequential_4/lstm_10/lstm_cell_10/Relu_1Relu+sequential_4/lstm_10/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Z╔
'sequential_4/lstm_10/lstm_cell_10/mul_2Mul/sequential_4/lstm_10/lstm_cell_10/Sigmoid_2:y:06sequential_4/lstm_10/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         ZЃ
2sequential_4/lstm_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   s
1sequential_4/lstm_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ё
$sequential_4/lstm_10/TensorArrayV2_1TensorListReserve;sequential_4/lstm_10/TensorArrayV2_1/element_shape:output:0:sequential_4/lstm_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм[
sequential_4/lstm_10/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-sequential_4/lstm_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         i
'sequential_4/lstm_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : е
sequential_4/lstm_10/whileWhile0sequential_4/lstm_10/while/loop_counter:output:06sequential_4/lstm_10/while/maximum_iterations:output:0"sequential_4/lstm_10/time:output:0-sequential_4/lstm_10/TensorArrayV2_1:handle:0#sequential_4/lstm_10/zeros:output:0%sequential_4/lstm_10/zeros_1:output:0-sequential_4/lstm_10/strided_slice_1:output:0Lsequential_4/lstm_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_4_lstm_10_lstm_cell_10_matmul_readvariableop_resourceBsequential_4_lstm_10_lstm_cell_10_matmul_1_readvariableop_resourceAsequential_4_lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&sequential_4_lstm_10_while_body_197371*2
cond*R(
&sequential_4_lstm_10_while_cond_197370*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations ќ
Esequential_4/lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   Ћ
7sequential_4/lstm_10/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_4/lstm_10/while:output:3Nsequential_4/lstm_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0*
num_elements}
*sequential_4/lstm_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         v
,sequential_4/lstm_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,sequential_4/lstm_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
$sequential_4/lstm_10/strided_slice_3StridedSlice@sequential_4/lstm_10/TensorArrayV2Stack/TensorListStack:tensor:03sequential_4/lstm_10/strided_slice_3/stack:output:05sequential_4/lstm_10/strided_slice_3/stack_1:output:05sequential_4/lstm_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maskz
%sequential_4/lstm_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
 sequential_4/lstm_10/transpose_1	Transpose@sequential_4/lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_4/lstm_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:         Zp
sequential_4/lstm_10/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ї
sequential_4/dropout_4/IdentityIdentity-sequential_4/lstm_10/strided_slice_3:output:0*
T0*'
_output_shapes
:         Zъ
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype0х
sequential_4/dense_4/MatMulMatMul(sequential_4/dropout_4/Identity:output:02sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ю
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0х
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         t
IdentityIdentity%sequential_4/dense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         Х
NoOpNoOp,^sequential_4/dense_4/BiasAdd/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp9^sequential_4/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp8^sequential_4/lstm_10/lstm_cell_10/MatMul/ReadVariableOp:^sequential_4/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp^sequential_4/lstm_10/while7^sequential_4/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp6^sequential_4/lstm_9/lstm_cell_9/MatMul/ReadVariableOp8^sequential_4/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp^sequential_4/lstm_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp2t
8sequential_4/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp8sequential_4/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp2r
7sequential_4/lstm_10/lstm_cell_10/MatMul/ReadVariableOp7sequential_4/lstm_10/lstm_cell_10/MatMul/ReadVariableOp2v
9sequential_4/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp9sequential_4/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp28
sequential_4/lstm_10/whilesequential_4/lstm_10/while2p
6sequential_4/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp6sequential_4/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp2n
5sequential_4/lstm_9/lstm_cell_9/MatMul/ReadVariableOp5sequential_4/lstm_9/lstm_cell_9/MatMul/ReadVariableOp2r
7sequential_4/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp7sequential_4/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp26
sequential_4/lstm_9/whilesequential_4/lstm_9/while:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_9_input
╚7
К
while_body_200064
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	УG
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:	ZУB
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	У
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	УE
2while_lstm_cell_9_matmul_1_readvariableop_resource:	ZУ@
1while_lstm_cell_9_biasadd_readvariableop_resource:	Уѕб(while/lstm_cell_9/BiasAdd/ReadVariableOpб'while/lstm_cell_9/MatMul/ReadVariableOpб)while/lstm_cell_9/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Џ
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	У*
dtype0И
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЪ
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0Ъ
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЏ
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         УЎ
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0ц
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уc
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :В
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitx
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Zz
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:         Zё
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         Zr
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:         ZЋ
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         Zі
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Zz
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:         Zo
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         ZЎ
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         Z─
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         Zx
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Z═

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
Д
Х
'__inference_lstm_9_layer_call_fn_199697
inputs_0
unknown:	У
	unknown_0:	ZУ
	unknown_1:	У
identityѕбStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_197804|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
╠K
Ю
C__inference_lstm_10_layer_call_and_return_conditional_losses_200625
inputs_0>
+lstm_cell_10_matmul_readvariableop_resource:	ZУ@
-lstm_cell_10_matmul_1_readvariableop_resource:	ZУ;
,lstm_cell_10_biasadd_readvariableop_resource:	У
identityѕб#lstm_cell_10/BiasAdd/ReadVariableOpб"lstm_cell_10/MatMul/ReadVariableOpб$lstm_cell_10/MatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maskЈ
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0ќ
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЊ
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0љ
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уї
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         УЇ
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0Ћ
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У^
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitn
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Zp
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:         Zw
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         Zh
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:         Zє
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         Z{
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Zp
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:         Ze
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Zі
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_200540*
condR
while_cond_200539*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         Z└
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  Z: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  Z
"
_user_specified_name
inputs_0
уI
ћ
B__inference_lstm_9_layer_call_and_return_conditional_losses_198875

inputs=
*lstm_cell_9_matmul_readvariableop_resource:	У?
,lstm_cell_9_matmul_1_readvariableop_resource:	ZУ:
+lstm_cell_9_biasadd_readvariableop_resource:	У
identityѕб"lstm_cell_9/BiasAdd/ReadVariableOpб!lstm_cell_9/MatMul/ReadVariableOpб#lstm_cell_9/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЇ
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	У*
dtype0ћ
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЉ
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0ј
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЅ
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         УІ
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0њ
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У]
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :┌
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitl
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Zn
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*'
_output_shapes
:         Zu
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         Zf
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*'
_output_shapes
:         ZЃ
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         Zx
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Zn
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*'
_output_shapes
:         Zc
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         ZЄ
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B :  
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_198791*
condR
while_cond_198790*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         Zй
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
в
Ш
-__inference_lstm_cell_10_layer_call_fn_201093

inputs
states_0
states_1
unknown:	ZУ
	unknown_0:	ZУ
	unknown_1:	У
identity

identity_1

identity_2ѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         Z:         Z:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_198028o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Zq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         Zq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         Z:         Z:         Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs:QM
'
_output_shapes
:         Z
"
_user_specified_name
states_0:QM
'
_output_shapes
:         Z
"
_user_specified_name
states_1
е9
л
while_body_200395
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	ZУH
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:	ZУC
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	У
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	ZУF
3while_lstm_cell_10_matmul_1_readvariableop_resource:	ZУA
2while_lstm_cell_10_biasadd_readvariableop_resource:	Уѕб)while/lstm_cell_10/BiasAdd/ReadVariableOpб(while/lstm_cell_10/MatMul/ReadVariableOpб*while/lstm_cell_10/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         Z*
element_dtype0Ю
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0║
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УА
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0А
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уъ
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         УЏ
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0Д
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уd
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitz
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Z|
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         Zє
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         Zt
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         Zў
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         ZЇ
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Z|
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         Zq
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Zю
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         Zr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ь
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_10/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: y
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         Zy
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Zл

while/NoOpNoOp*^while/lstm_cell_10/BiasAdd/ReadVariableOp)^while/lstm_cell_10/MatMul/ReadVariableOp+^while/lstm_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2V
)while/lstm_cell_10/BiasAdd/ReadVariableOp)while/lstm_cell_10/BiasAdd/ReadVariableOp2T
(while/lstm_cell_10/MatMul/ReadVariableOp(while/lstm_cell_10/MatMul/ReadVariableOp2X
*while/lstm_cell_10/MatMul_1/ReadVariableOp*while/lstm_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
ж
ш
,__inference_lstm_cell_9_layer_call_fn_200995

inputs
states_0
states_1
unknown:	У
	unknown_0:	ZУ
	unknown_1:	У
identity

identity_1

identity_2ѕбStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         Z:         Z:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_197676o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Zq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         Zq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         Z:         Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         Z
"
_user_specified_name
states_0:QM
'
_output_shapes
:         Z
"
_user_specified_name
states_1
Т
М
%sequential_4_lstm_9_while_cond_197230D
@sequential_4_lstm_9_while_sequential_4_lstm_9_while_loop_counterJ
Fsequential_4_lstm_9_while_sequential_4_lstm_9_while_maximum_iterations)
%sequential_4_lstm_9_while_placeholder+
'sequential_4_lstm_9_while_placeholder_1+
'sequential_4_lstm_9_while_placeholder_2+
'sequential_4_lstm_9_while_placeholder_3F
Bsequential_4_lstm_9_while_less_sequential_4_lstm_9_strided_slice_1\
Xsequential_4_lstm_9_while_sequential_4_lstm_9_while_cond_197230___redundant_placeholder0\
Xsequential_4_lstm_9_while_sequential_4_lstm_9_while_cond_197230___redundant_placeholder1\
Xsequential_4_lstm_9_while_sequential_4_lstm_9_while_cond_197230___redundant_placeholder2\
Xsequential_4_lstm_9_while_sequential_4_lstm_9_while_cond_197230___redundant_placeholder3&
"sequential_4_lstm_9_while_identity
▓
sequential_4/lstm_9/while/LessLess%sequential_4_lstm_9_while_placeholderBsequential_4_lstm_9_while_less_sequential_4_lstm_9_strided_slice_1*
T0*
_output_shapes
: s
"sequential_4/lstm_9/while/IdentityIdentity"sequential_4/lstm_9/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_4_lstm_9_while_identity+sequential_4/lstm_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
х
├
while_cond_197734
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_197734___redundant_placeholder04
0while_while_cond_197734___redundant_placeholder14
0while_while_cond_197734___redundant_placeholder24
0while_while_cond_197734___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
Ј
и
(__inference_lstm_10_layer_call_fn_200302
inputs_0
unknown:	ZУ
	unknown_0:	ZУ
	unknown_1:	У
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_10_layer_call_and_return_conditional_losses_197965o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  Z
"
_user_specified_name
inputs_0
х
├
while_cond_197894
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_197894___redundant_placeholder04
0while_while_cond_197894___redundant_placeholder14
0while_while_cond_197894___redundant_placeholder24
0while_while_cond_197894___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
э
х
(__inference_lstm_10_layer_call_fn_200335

inputs
unknown:	ZУ
	unknown_0:	ZУ
	unknown_1:	У
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_10_layer_call_and_return_conditional_losses_198710o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         Z: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         Z
 
_user_specified_nameinputs
┐
┘
H__inference_sequential_4_layer_call_and_return_conditional_losses_198995
lstm_9_input 
lstm_9_198974:	У 
lstm_9_198976:	ZУ
lstm_9_198978:	У!
lstm_10_198981:	ZУ!
lstm_10_198983:	ZУ
lstm_10_198985:	У 
dense_4_198989:Z
dense_4_198991:
identityѕбdense_4/StatefulPartitionedCallбlstm_10/StatefulPartitionedCallбlstm_9/StatefulPartitionedCallЃ
lstm_9/StatefulPartitionedCallStatefulPartitionedCalllstm_9_inputlstm_9_198974lstm_9_198976lstm_9_198978*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_198316Ъ
lstm_10/StatefulPartitionedCallStatefulPartitionedCall'lstm_9/StatefulPartitionedCall:output:0lstm_10_198981lstm_10_198983lstm_10_198985*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_10_layer_call_and_return_conditional_losses_198468▄
dropout_4/PartitionedCallPartitionedCall(lstm_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_198481ѕ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_4_198989dense_4_198991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_198493w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ф
NoOpNoOp ^dense_4/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_9_input
Г
М
H__inference_sequential_4_layer_call_and_return_conditional_losses_198500

inputs 
lstm_9_198317:	У 
lstm_9_198319:	ZУ
lstm_9_198321:	У!
lstm_10_198469:	ZУ!
lstm_10_198471:	ZУ
lstm_10_198473:	У 
dense_4_198494:Z
dense_4_198496:
identityѕбdense_4/StatefulPartitionedCallбlstm_10/StatefulPartitionedCallбlstm_9/StatefulPartitionedCall§
lstm_9/StatefulPartitionedCallStatefulPartitionedCallinputslstm_9_198317lstm_9_198319lstm_9_198321*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_198316Ъ
lstm_10/StatefulPartitionedCallStatefulPartitionedCall'lstm_9/StatefulPartitionedCall:output:0lstm_10_198469lstm_10_198471lstm_10_198473*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_10_layer_call_and_return_conditional_losses_198468▄
dropout_4/PartitionedCallPartitionedCall(lstm_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_198481ѕ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_4_198494dense_4_198496*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_198493w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Ф
NoOpNoOp ^dense_4/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Тѓ
Б
"__inference__traced_restore_201370
file_prefix1
assignvariableop_dense_4_kernel:Z-
assignvariableop_1_dense_4_bias:?
,assignvariableop_2_lstm_9_lstm_cell_9_kernel:	УI
6assignvariableop_3_lstm_9_lstm_cell_9_recurrent_kernel:	ZУ9
*assignvariableop_4_lstm_9_lstm_cell_9_bias:	УA
.assignvariableop_5_lstm_10_lstm_cell_10_kernel:	ZУK
8assignvariableop_6_lstm_10_lstm_cell_10_recurrent_kernel:	ZУ;
,assignvariableop_7_lstm_10_lstm_cell_10_bias:	У&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: G
4assignvariableop_10_adam_m_lstm_9_lstm_cell_9_kernel:	УG
4assignvariableop_11_adam_v_lstm_9_lstm_cell_9_kernel:	УQ
>assignvariableop_12_adam_m_lstm_9_lstm_cell_9_recurrent_kernel:	ZУQ
>assignvariableop_13_adam_v_lstm_9_lstm_cell_9_recurrent_kernel:	ZУA
2assignvariableop_14_adam_m_lstm_9_lstm_cell_9_bias:	УA
2assignvariableop_15_adam_v_lstm_9_lstm_cell_9_bias:	УI
6assignvariableop_16_adam_m_lstm_10_lstm_cell_10_kernel:	ZУI
6assignvariableop_17_adam_v_lstm_10_lstm_cell_10_kernel:	ZУS
@assignvariableop_18_adam_m_lstm_10_lstm_cell_10_recurrent_kernel:	ZУS
@assignvariableop_19_adam_v_lstm_10_lstm_cell_10_recurrent_kernel:	ZУC
4assignvariableop_20_adam_m_lstm_10_lstm_cell_10_bias:	УC
4assignvariableop_21_adam_v_lstm_10_lstm_cell_10_bias:	У;
)assignvariableop_22_adam_m_dense_4_kernel:Z;
)assignvariableop_23_adam_v_dense_4_kernel:Z5
'assignvariableop_24_adam_m_dense_4_bias:5
'assignvariableop_25_adam_v_dense_4_bias:%
assignvariableop_26_total_1: %
assignvariableop_27_count_1: #
assignvariableop_28_total: #
assignvariableop_29_count: 
identity_31ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9р
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Є
value§BЩB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH«
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ║
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*љ
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOpAssignVariableOpassignvariableop_dense_4_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_2AssignVariableOp,assignvariableop_2_lstm_9_lstm_cell_9_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_3AssignVariableOp6assignvariableop_3_lstm_9_lstm_cell_9_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_4AssignVariableOp*assignvariableop_4_lstm_9_lstm_cell_9_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_5AssignVariableOp.assignvariableop_5_lstm_10_lstm_cell_10_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_6AssignVariableOp8assignvariableop_6_lstm_10_lstm_cell_10_recurrent_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_10_lstm_cell_10_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:│
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_10AssignVariableOp4assignvariableop_10_adam_m_lstm_9_lstm_cell_9_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_11AssignVariableOp4assignvariableop_11_adam_v_lstm_9_lstm_cell_9_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_12AssignVariableOp>assignvariableop_12_adam_m_lstm_9_lstm_cell_9_recurrent_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_13AssignVariableOp>assignvariableop_13_adam_v_lstm_9_lstm_cell_9_recurrent_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:╦
AssignVariableOp_14AssignVariableOp2assignvariableop_14_adam_m_lstm_9_lstm_cell_9_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:╦
AssignVariableOp_15AssignVariableOp2assignvariableop_15_adam_v_lstm_9_lstm_cell_9_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_16AssignVariableOp6assignvariableop_16_adam_m_lstm_10_lstm_cell_10_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_v_lstm_10_lstm_cell_10_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_18AssignVariableOp@assignvariableop_18_adam_m_lstm_10_lstm_cell_10_recurrent_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_19AssignVariableOp@assignvariableop_19_adam_v_lstm_10_lstm_cell_10_recurrent_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_m_lstm_10_lstm_cell_10_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_v_lstm_10_lstm_cell_10_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_m_dense_4_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_v_dense_4_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_m_dense_4_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_v_dense_4_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_1Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 с
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: л
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
х
├
while_cond_198231
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_198231___redundant_placeholder04
0while_while_cond_198231___redundant_placeholder14
0while_while_cond_198231___redundant_placeholder24
0while_while_cond_198231___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
уI
ћ
B__inference_lstm_9_layer_call_and_return_conditional_losses_198316

inputs=
*lstm_cell_9_matmul_readvariableop_resource:	У?
,lstm_cell_9_matmul_1_readvariableop_resource:	ZУ:
+lstm_cell_9_biasadd_readvariableop_resource:	У
identityѕб"lstm_cell_9/BiasAdd/ReadVariableOpб!lstm_cell_9/MatMul/ReadVariableOpб#lstm_cell_9/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЇ
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	У*
dtype0ћ
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЉ
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0ј
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЅ
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         УІ
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0њ
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У]
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :┌
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitl
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Zn
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*'
_output_shapes
:         Zu
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         Zf
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*'
_output_shapes
:         ZЃ
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         Zx
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Zn
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*'
_output_shapes
:         Zc
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         ZЄ
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B :  
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_198232*
condR
while_cond_198231*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         Zй
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
х
├
while_cond_200539
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_200539___redundant_placeholder04
0while_while_cond_200539___redundant_placeholder14
0while_while_cond_200539___redundant_placeholder24
0while_while_cond_200539___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
н
Ѓ
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_197530

inputs

states
states_11
matmul_readvariableop_resource:	У3
 matmul_1_readvariableop_resource:	ZУ.
biasadd_readvariableop_resource:	У
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	У*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Уs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         ZV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         ZU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         ZN
ReluRelusplit:output:2*
T0*'
_output_shapes
:         Z_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         ZT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         ZV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         ZK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         Zc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         ZX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         ZZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         ZZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         ZЉ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         Z:         Z: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         Z
 
_user_specified_namestates:OK
'
_output_shapes
:         Z
 
_user_specified_namestates
нB
║
__inference__traced_save_201270
file_prefix-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop8
4savev2_lstm_9_lstm_cell_9_kernel_read_readvariableopB
>savev2_lstm_9_lstm_cell_9_recurrent_kernel_read_readvariableop6
2savev2_lstm_9_lstm_cell_9_bias_read_readvariableop:
6savev2_lstm_10_lstm_cell_10_kernel_read_readvariableopD
@savev2_lstm_10_lstm_cell_10_recurrent_kernel_read_readvariableop8
4savev2_lstm_10_lstm_cell_10_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop?
;savev2_adam_m_lstm_9_lstm_cell_9_kernel_read_readvariableop?
;savev2_adam_v_lstm_9_lstm_cell_9_kernel_read_readvariableopI
Esavev2_adam_m_lstm_9_lstm_cell_9_recurrent_kernel_read_readvariableopI
Esavev2_adam_v_lstm_9_lstm_cell_9_recurrent_kernel_read_readvariableop=
9savev2_adam_m_lstm_9_lstm_cell_9_bias_read_readvariableop=
9savev2_adam_v_lstm_9_lstm_cell_9_bias_read_readvariableopA
=savev2_adam_m_lstm_10_lstm_cell_10_kernel_read_readvariableopA
=savev2_adam_v_lstm_10_lstm_cell_10_kernel_read_readvariableopK
Gsavev2_adam_m_lstm_10_lstm_cell_10_recurrent_kernel_read_readvariableopK
Gsavev2_adam_v_lstm_10_lstm_cell_10_recurrent_kernel_read_readvariableop?
;savev2_adam_m_lstm_10_lstm_cell_10_bias_read_readvariableop?
;savev2_adam_v_lstm_10_lstm_cell_10_bias_read_readvariableop4
0savev2_adam_m_dense_4_kernel_read_readvariableop4
0savev2_adam_v_dense_4_kernel_read_readvariableop2
.savev2_adam_m_dense_4_bias_read_readvariableop2
.savev2_adam_v_dense_4_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partЂ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: я
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Є
value§BЩB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHФ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ╠
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop4savev2_lstm_9_lstm_cell_9_kernel_read_readvariableop>savev2_lstm_9_lstm_cell_9_recurrent_kernel_read_readvariableop2savev2_lstm_9_lstm_cell_9_bias_read_readvariableop6savev2_lstm_10_lstm_cell_10_kernel_read_readvariableop@savev2_lstm_10_lstm_cell_10_recurrent_kernel_read_readvariableop4savev2_lstm_10_lstm_cell_10_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop;savev2_adam_m_lstm_9_lstm_cell_9_kernel_read_readvariableop;savev2_adam_v_lstm_9_lstm_cell_9_kernel_read_readvariableopEsavev2_adam_m_lstm_9_lstm_cell_9_recurrent_kernel_read_readvariableopEsavev2_adam_v_lstm_9_lstm_cell_9_recurrent_kernel_read_readvariableop9savev2_adam_m_lstm_9_lstm_cell_9_bias_read_readvariableop9savev2_adam_v_lstm_9_lstm_cell_9_bias_read_readvariableop=savev2_adam_m_lstm_10_lstm_cell_10_kernel_read_readvariableop=savev2_adam_v_lstm_10_lstm_cell_10_kernel_read_readvariableopGsavev2_adam_m_lstm_10_lstm_cell_10_recurrent_kernel_read_readvariableopGsavev2_adam_v_lstm_10_lstm_cell_10_recurrent_kernel_read_readvariableop;savev2_adam_m_lstm_10_lstm_cell_10_bias_read_readvariableop;savev2_adam_v_lstm_10_lstm_cell_10_bias_read_readvariableop0savev2_adam_m_dense_4_kernel_read_readvariableop0savev2_adam_v_dense_4_kernel_read_readvariableop.savev2_adam_m_dense_4_bias_read_readvariableop.savev2_adam_v_dense_4_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *-
dtypes#
!2	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ѓ
_input_shapesы
Ь: :Z::	У:	ZУ:У:	ZУ:	ZУ:У: : :	У:	У:	ZУ:	ZУ:У:У:	ZУ:	ZУ:	ZУ:	ZУ:У:У:Z:Z::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:Z: 

_output_shapes
::%!

_output_shapes
:	У:%!

_output_shapes
:	ZУ:!

_output_shapes	
:У:%!

_output_shapes
:	ZУ:%!

_output_shapes
:	ZУ:!

_output_shapes	
:У:	

_output_shapes
: :


_output_shapes
: :%!

_output_shapes
:	У:%!

_output_shapes
:	У:%!

_output_shapes
:	ZУ:%!

_output_shapes
:	ZУ:!

_output_shapes	
:У:!

_output_shapes	
:У:%!

_output_shapes
:	ZУ:%!

_output_shapes
:	ZУ:%!

_output_shapes
:	ZУ:%!

_output_shapes
:	ZУ:!

_output_shapes	
:У:!

_output_shapes	
:У:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ђ┤
Ь
H__inference_sequential_4_layer_call_and_return_conditional_losses_199675

inputsD
1lstm_9_lstm_cell_9_matmul_readvariableop_resource:	УF
3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource:	ZУA
2lstm_9_lstm_cell_9_biasadd_readvariableop_resource:	УF
3lstm_10_lstm_cell_10_matmul_readvariableop_resource:	ZУH
5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource:	ZУC
4lstm_10_lstm_cell_10_biasadd_readvariableop_resource:	У8
&dense_4_matmul_readvariableop_resource:Z5
'dense_4_biasadd_readvariableop_resource:
identityѕбdense_4/BiasAdd/ReadVariableOpбdense_4/MatMul/ReadVariableOpб+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpб*lstm_10/lstm_cell_10/MatMul/ReadVariableOpб,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpбlstm_10/whileб)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpб(lstm_9/lstm_cell_9/MatMul/ReadVariableOpб*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpбlstm_9/whileB
lstm_9/ShapeShapeinputs*
T0*
_output_shapes
:d
lstm_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
lstm_9/strided_sliceStridedSlicelstm_9/Shape:output:0#lstm_9/strided_slice/stack:output:0%lstm_9/strided_slice/stack_1:output:0%lstm_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zѕ
lstm_9/zeros/packedPacklstm_9/strided_slice:output:0lstm_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ђ
lstm_9/zerosFilllstm_9/zeros/packed:output:0lstm_9/zeros/Const:output:0*
T0*'
_output_shapes
:         ZY
lstm_9/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zї
lstm_9/zeros_1/packedPacklstm_9/strided_slice:output:0 lstm_9/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_9/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Є
lstm_9/zeros_1Filllstm_9/zeros_1/packed:output:0lstm_9/zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zj
lstm_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          {
lstm_9/transpose	Transposeinputslstm_9/transpose/perm:output:0*
T0*+
_output_shapes
:         R
lstm_9/Shape_1Shapelstm_9/transpose:y:0*
T0*
_output_shapes
:f
lstm_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:■
lstm_9/strided_slice_1StridedSlicelstm_9/Shape_1:output:0%lstm_9/strided_slice_1/stack:output:0'lstm_9/strided_slice_1/stack_1:output:0'lstm_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
lstm_9/TensorArrayV2TensorListReserve+lstm_9/TensorArrayV2/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмЇ
<lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ш
.lstm_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_9/transpose:y:0Elstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмf
lstm_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ї
lstm_9/strided_slice_2StridedSlicelstm_9/transpose:y:0%lstm_9/strided_slice_2/stack:output:0'lstm_9/strided_slice_2/stack_1:output:0'lstm_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЏ
(lstm_9/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp1lstm_9_lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	У*
dtype0Е
lstm_9/lstm_cell_9/MatMulMatMullstm_9/strided_slice_2:output:00lstm_9/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЪ
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0Б
lstm_9/lstm_cell_9/MatMul_1MatMullstm_9/zeros:output:02lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уъ
lstm_9/lstm_cell_9/addAddV2#lstm_9/lstm_cell_9/MatMul:product:0%lstm_9/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:         УЎ
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp2lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0Д
lstm_9/lstm_cell_9/BiasAddBiasAddlstm_9/lstm_cell_9/add:z:01lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уd
"lstm_9/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :№
lstm_9/lstm_cell_9/splitSplit+lstm_9/lstm_cell_9/split/split_dim:output:0#lstm_9/lstm_cell_9/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitz
lstm_9/lstm_cell_9/SigmoidSigmoid!lstm_9/lstm_cell_9/split:output:0*
T0*'
_output_shapes
:         Z|
lstm_9/lstm_cell_9/Sigmoid_1Sigmoid!lstm_9/lstm_cell_9/split:output:1*
T0*'
_output_shapes
:         Zі
lstm_9/lstm_cell_9/mulMul lstm_9/lstm_cell_9/Sigmoid_1:y:0lstm_9/zeros_1:output:0*
T0*'
_output_shapes
:         Zt
lstm_9/lstm_cell_9/ReluRelu!lstm_9/lstm_cell_9/split:output:2*
T0*'
_output_shapes
:         Zў
lstm_9/lstm_cell_9/mul_1Mullstm_9/lstm_cell_9/Sigmoid:y:0%lstm_9/lstm_cell_9/Relu:activations:0*
T0*'
_output_shapes
:         ZЇ
lstm_9/lstm_cell_9/add_1AddV2lstm_9/lstm_cell_9/mul:z:0lstm_9/lstm_cell_9/mul_1:z:0*
T0*'
_output_shapes
:         Z|
lstm_9/lstm_cell_9/Sigmoid_2Sigmoid!lstm_9/lstm_cell_9/split:output:3*
T0*'
_output_shapes
:         Zq
lstm_9/lstm_cell_9/Relu_1Relulstm_9/lstm_cell_9/add_1:z:0*
T0*'
_output_shapes
:         Zю
lstm_9/lstm_cell_9/mul_2Mul lstm_9/lstm_cell_9/Sigmoid_2:y:0'lstm_9/lstm_cell_9/Relu_1:activations:0*
T0*'
_output_shapes
:         Zu
$lstm_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ═
lstm_9/TensorArrayV2_1TensorListReserve-lstm_9/TensorArrayV2_1/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмM
lstm_9/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         [
lstm_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : р
lstm_9/whileWhile"lstm_9/while/loop_counter:output:0(lstm_9/while/maximum_iterations:output:0lstm_9/time:output:0lstm_9/TensorArrayV2_1:handle:0lstm_9/zeros:output:0lstm_9/zeros_1:output:0lstm_9/strided_slice_1:output:0>lstm_9/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_9_lstm_cell_9_matmul_readvariableop_resource3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource2lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_9_while_body_199436*$
condR
lstm_9_while_cond_199435*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations ѕ
7lstm_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   О
)lstm_9/TensorArrayV2Stack/TensorListStackTensorListStacklstm_9/while:output:3@lstm_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0o
lstm_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         h
lstm_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
lstm_9/strided_slice_3StridedSlice2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_9/strided_slice_3/stack:output:0'lstm_9/strided_slice_3/stack_1:output:0'lstm_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maskl
lstm_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ф
lstm_9/transpose_1	Transpose2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:         Zb
lstm_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    S
lstm_10/ShapeShapelstm_9/transpose_1:y:0*
T0*
_output_shapes
:e
lstm_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_10/strided_sliceStridedSlicelstm_10/Shape:output:0$lstm_10/strided_slice/stack:output:0&lstm_10/strided_slice/stack_1:output:0&lstm_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ZІ
lstm_10/zeros/packedPacklstm_10/strided_slice:output:0lstm_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ё
lstm_10/zerosFilllstm_10/zeros/packed:output:0lstm_10/zeros/Const:output:0*
T0*'
_output_shapes
:         ZZ
lstm_10/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ZЈ
lstm_10/zeros_1/packedPacklstm_10/strided_slice:output:0!lstm_10/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_10/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    і
lstm_10/zeros_1Filllstm_10/zeros_1/packed:output:0lstm_10/zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zk
lstm_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ї
lstm_10/transpose	Transposelstm_9/transpose_1:y:0lstm_10/transpose/perm:output:0*
T0*+
_output_shapes
:         ZT
lstm_10/Shape_1Shapelstm_10/transpose:y:0*
T0*
_output_shapes
:g
lstm_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
lstm_10/strided_slice_1StridedSlicelstm_10/Shape_1:output:0&lstm_10/strided_slice_1/stack:output:0(lstm_10/strided_slice_1/stack_1:output:0(lstm_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_10/TensorArrayV2TensorListReserve,lstm_10/TensorArrayV2/element_shape:output:0 lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмј
=lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   Э
/lstm_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_10/transpose:y:0Flstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмg
lstm_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
lstm_10/strided_slice_2StridedSlicelstm_10/transpose:y:0&lstm_10/strided_slice_2/stack:output:0(lstm_10/strided_slice_2/stack_1:output:0(lstm_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maskЪ
*lstm_10/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3lstm_10_lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0«
lstm_10/lstm_cell_10/MatMulMatMul lstm_10/strided_slice_2:output:02lstm_10/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УБ
,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0е
lstm_10/lstm_cell_10/MatMul_1MatMullstm_10/zeros:output:04lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уц
lstm_10/lstm_cell_10/addAddV2%lstm_10/lstm_cell_10/MatMul:product:0'lstm_10/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         УЮ
+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0Г
lstm_10/lstm_cell_10/BiasAddBiasAddlstm_10/lstm_cell_10/add:z:03lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уf
$lstm_10/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
lstm_10/lstm_cell_10/splitSplit-lstm_10/lstm_cell_10/split/split_dim:output:0%lstm_10/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_split~
lstm_10/lstm_cell_10/SigmoidSigmoid#lstm_10/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Zђ
lstm_10/lstm_cell_10/Sigmoid_1Sigmoid#lstm_10/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         ZЈ
lstm_10/lstm_cell_10/mulMul"lstm_10/lstm_cell_10/Sigmoid_1:y:0lstm_10/zeros_1:output:0*
T0*'
_output_shapes
:         Zx
lstm_10/lstm_cell_10/ReluRelu#lstm_10/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         Zъ
lstm_10/lstm_cell_10/mul_1Mul lstm_10/lstm_cell_10/Sigmoid:y:0'lstm_10/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         ZЊ
lstm_10/lstm_cell_10/add_1AddV2lstm_10/lstm_cell_10/mul:z:0lstm_10/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Zђ
lstm_10/lstm_cell_10/Sigmoid_2Sigmoid#lstm_10/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         Zu
lstm_10/lstm_cell_10/Relu_1Relulstm_10/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Zб
lstm_10/lstm_cell_10/mul_2Mul"lstm_10/lstm_cell_10/Sigmoid_2:y:0)lstm_10/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         Zv
%lstm_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   f
$lstm_10/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_10/TensorArrayV2_1TensorListReserve.lstm_10/TensorArrayV2_1/element_shape:output:0-lstm_10/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмN
lstm_10/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ы
lstm_10/whileWhile#lstm_10/while/loop_counter:output:0)lstm_10/while/maximum_iterations:output:0lstm_10/time:output:0 lstm_10/TensorArrayV2_1:handle:0lstm_10/zeros:output:0lstm_10/zeros_1:output:0 lstm_10/strided_slice_1:output:0?lstm_10/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_10_lstm_cell_10_matmul_readvariableop_resource5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource4lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_10_while_body_199576*%
condR
lstm_10_while_cond_199575*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ѕ
8lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   Ь
*lstm_10/TensorArrayV2Stack/TensorListStackTensorListStacklstm_10/while:output:3Alstm_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0*
num_elementsp
lstm_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
lstm_10/strided_slice_3StridedSlice3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_10/strided_slice_3/stack:output:0(lstm_10/strided_slice_3/stack_1:output:0(lstm_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maskm
lstm_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_10/transpose_1	Transpose3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:         Zc
lstm_10/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    \
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?њ
dropout_4/dropout/MulMul lstm_10/strided_slice_3:output:0 dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:         Zg
dropout_4/dropout/ShapeShape lstm_10/strided_slice_3:output:0*
T0*
_output_shapes
:а
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:         Z*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>─
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Z^
dropout_4/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ╗
dropout_4/dropout/SelectV2SelectV2"dropout_4/dropout/GreaterEqual:z:0dropout_4/dropout/Mul:z:0"dropout_4/dropout/Const_1:output:0*
T0*'
_output_shapes
:         Zё
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype0ќ
dense_4/MatMulMatMul#dropout_4/dropout/SelectV2:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ѓ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ј
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         g
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ┤
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp,^lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp+^lstm_10/lstm_cell_10/MatMul/ReadVariableOp-^lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp^lstm_10/while*^lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp)^lstm_9/lstm_cell_9/MatMul/ReadVariableOp+^lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp^lstm_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2Z
+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp2X
*lstm_10/lstm_cell_10/MatMul/ReadVariableOp*lstm_10/lstm_cell_10/MatMul/ReadVariableOp2\
,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp2
lstm_10/whilelstm_10/while2V
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp2T
(lstm_9/lstm_cell_9/MatMul/ReadVariableOp(lstm_9/lstm_cell_9/MatMul/ReadVariableOp2X
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp2
lstm_9/whilelstm_9/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╬"
▄
while_body_197544
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_9_197568_0:	У-
while_lstm_cell_9_197570_0:	ZУ)
while_lstm_cell_9_197572_0:	У
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_9_197568:	У+
while_lstm_cell_9_197570:	ZУ'
while_lstm_cell_9_197572:	Уѕб)while/lstm_cell_9/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0«
)while/lstm_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_9_197568_0while_lstm_cell_9_197570_0while_lstm_cell_9_197572_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         Z:         Z:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_197530█
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_9/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ј
while/Identity_4Identity2while/lstm_cell_9/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         ZЈ
while/Identity_5Identity2while/lstm_cell_9/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         Zx

while/NoOpNoOp*^while/lstm_cell_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_9_197568while_lstm_cell_9_197568_0"6
while_lstm_cell_9_197570while_lstm_cell_9_197570_0"6
while_lstm_cell_9_197572while_lstm_cell_9_197572_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2V
)while/lstm_cell_9/StatefulPartitionedCall)while/lstm_cell_9/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
к	
З
C__inference_dense_4_layer_call_and_return_conditional_losses_200961

inputs0
matmul_readvariableop_resource:Z-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
х
├
while_cond_198790
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_198790___redundant_placeholder04
0while_while_cond_198790___redundant_placeholder14
0while_while_cond_198790___redundant_placeholder24
0while_while_cond_198790___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
ЕK
Џ
C__inference_lstm_10_layer_call_and_return_conditional_losses_200770

inputs>
+lstm_cell_10_matmul_readvariableop_resource:	ZУ@
-lstm_cell_10_matmul_1_readvariableop_resource:	ZУ;
,lstm_cell_10_biasadd_readvariableop_resource:	У
identityѕб#lstm_cell_10/BiasAdd/ReadVariableOpб"lstm_cell_10/MatMul/ReadVariableOpб$lstm_cell_10/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maskЈ
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0ќ
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЊ
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0љ
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уї
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         УЇ
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0Ћ
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У^
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitn
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Zp
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:         Zw
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         Zh
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:         Zє
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         Z{
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Zp
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:         Ze
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Zі
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_200685*
condR
while_cond_200684*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         Z└
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         Z: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         Z
 
_user_specified_nameinputs
х
├
while_cond_200829
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_200829___redundant_placeholder04
0while_while_cond_200829___redundant_placeholder14
0while_while_cond_200829___redundant_placeholder24
0while_while_cond_200829___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
Д
Х
'__inference_lstm_9_layer_call_fn_199686
inputs_0
unknown:	У
	unknown_0:	ZУ
	unknown_1:	У
identityѕбStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_lstm_9_layer_call_and_return_conditional_losses_197613|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
х
├
while_cond_200206
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_200206___redundant_placeholder04
0while_while_cond_200206___redundant_placeholder14
0while_while_cond_200206___redundant_placeholder24
0while_while_cond_200206___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
ЕK
Џ
C__inference_lstm_10_layer_call_and_return_conditional_losses_198468

inputs>
+lstm_cell_10_matmul_readvariableop_resource:	ZУ@
-lstm_cell_10_matmul_1_readvariableop_resource:	ZУ;
,lstm_cell_10_biasadd_readvariableop_resource:	У
identityѕб#lstm_cell_10/BiasAdd/ReadVariableOpб"lstm_cell_10/MatMul/ReadVariableOpб$lstm_cell_10/MatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maskЈ
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0ќ
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЊ
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0љ
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уї
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         УЇ
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0Ћ
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У^
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitn
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Zp
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:         Zw
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         Zh
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:         Zє
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         Z{
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Zp
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:         Ze
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Zі
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_198383*
condR
while_cond_198382*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         Z└
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         Z: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         Z
 
_user_specified_nameinputs
х
├
while_cond_200684
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_200684___redundant_placeholder04
0while_while_cond_200684___redundant_placeholder14
0while_while_cond_200684___redundant_placeholder24
0while_while_cond_200684___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
╠K
Ю
C__inference_lstm_10_layer_call_and_return_conditional_losses_200480
inputs_0>
+lstm_cell_10_matmul_readvariableop_resource:	ZУ@
-lstm_cell_10_matmul_1_readvariableop_resource:	ZУ;
,lstm_cell_10_biasadd_readvariableop_resource:	У
identityѕб#lstm_cell_10/BiasAdd/ReadVariableOpб"lstm_cell_10/MatMul/ReadVariableOpб$lstm_cell_10/MatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maskЈ
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0ќ
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УЊ
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	ZУ*
dtype0љ
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уї
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         УЇ
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:У*
dtype0Ћ
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У^
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :П
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitn
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Zp
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:         Zw
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         Zh
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:         Zє
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         Z{
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Zp
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:         Ze
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Zі
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         Zn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ѓ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_200395*
condR
while_cond_200394*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         Z└
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  Z: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  Z
"
_user_specified_name
inputs_0
п	
к
-__inference_sequential_4_layer_call_fn_199086

inputs
unknown:	У
	unknown_0:	ZУ
	unknown_1:	У
	unknown_2:	ZУ
	unknown_3:	ZУ
	unknown_4:	У
	unknown_5:Z
	unknown_6:
identityѕбStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_198931o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
§8
Ё
C__inference_lstm_10_layer_call_and_return_conditional_losses_197965

inputs&
lstm_cell_10_197881:	ZУ&
lstm_cell_10_197883:	ZУ"
lstm_cell_10_197885:	У
identityѕб$lstm_cell_10/StatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  ZD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maskш
$lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10_197881lstm_cell_10_197883lstm_cell_10_197885*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         Z:         Z:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_197880n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10_197881lstm_cell_10_197883lstm_cell_10_197885*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_197895*
condR
while_cond_197894*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         Z*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         Zu
NoOpNoOp%^lstm_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  Z: : : 2L
$lstm_cell_10/StatefulPartitionedCall$lstm_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  Z
 
_user_specified_nameinputs
жB
л

lstm_10_while_body_199285,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3+
'lstm_10_while_lstm_10_strided_slice_1_0g
clstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0:	ZУP
=lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0:	ZУK
<lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0:	У
lstm_10_while_identity
lstm_10_while_identity_1
lstm_10_while_identity_2
lstm_10_while_identity_3
lstm_10_while_identity_4
lstm_10_while_identity_5)
%lstm_10_while_lstm_10_strided_slice_1e
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorL
9lstm_10_while_lstm_cell_10_matmul_readvariableop_resource:	ZУN
;lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource:	ZУI
:lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource:	Уѕб1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpб0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpб2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpљ
?lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ╬
1lstm_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0lstm_10_while_placeholderHlstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         Z*
element_dtype0Г
0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0м
!lstm_10/while/lstm_cell_10/MatMulMatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У▒
2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp=lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0╣
#lstm_10/while/lstm_cell_10/MatMul_1MatMullstm_10_while_placeholder_2:lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УХ
lstm_10/while/lstm_cell_10/addAddV2+lstm_10/while/lstm_cell_10/MatMul:product:0-lstm_10/while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         УФ
1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp<lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0┐
"lstm_10/while/lstm_cell_10/BiasAddBiasAdd"lstm_10/while/lstm_cell_10/add:z:09lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уl
*lstm_10/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Є
 lstm_10/while/lstm_cell_10/splitSplit3lstm_10/while/lstm_cell_10/split/split_dim:output:0+lstm_10/while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitі
"lstm_10/while/lstm_cell_10/SigmoidSigmoid)lstm_10/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Zї
$lstm_10/while/lstm_cell_10/Sigmoid_1Sigmoid)lstm_10/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         Zъ
lstm_10/while/lstm_cell_10/mulMul(lstm_10/while/lstm_cell_10/Sigmoid_1:y:0lstm_10_while_placeholder_3*
T0*'
_output_shapes
:         Zё
lstm_10/while/lstm_cell_10/ReluRelu)lstm_10/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         Z░
 lstm_10/while/lstm_cell_10/mul_1Mul&lstm_10/while/lstm_cell_10/Sigmoid:y:0-lstm_10/while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         ZЦ
 lstm_10/while/lstm_cell_10/add_1AddV2"lstm_10/while/lstm_cell_10/mul:z:0$lstm_10/while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Zї
$lstm_10/while/lstm_cell_10/Sigmoid_2Sigmoid)lstm_10/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         ZЂ
!lstm_10/while/lstm_cell_10/Relu_1Relu$lstm_10/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Z┤
 lstm_10/while/lstm_cell_10/mul_2Mul(lstm_10/while/lstm_cell_10/Sigmoid_2:y:0/lstm_10/while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         Zz
8lstm_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ї
2lstm_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_10_while_placeholder_1Alstm_10/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_10/while/lstm_cell_10/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмU
lstm_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_10/while/addAddV2lstm_10_while_placeholderlstm_10/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Є
lstm_10/while/add_1AddV2(lstm_10_while_lstm_10_while_loop_counterlstm_10/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_10/while/IdentityIdentitylstm_10/while/add_1:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: і
lstm_10/while/Identity_1Identity.lstm_10_while_lstm_10_while_maximum_iterations^lstm_10/while/NoOp*
T0*
_output_shapes
: q
lstm_10/while/Identity_2Identitylstm_10/while/add:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: ъ
lstm_10/while/Identity_3IdentityBlstm_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_10/while/NoOp*
T0*
_output_shapes
: Љ
lstm_10/while/Identity_4Identity$lstm_10/while/lstm_cell_10/mul_2:z:0^lstm_10/while/NoOp*
T0*'
_output_shapes
:         ZЉ
lstm_10/while/Identity_5Identity$lstm_10/while/lstm_cell_10/add_1:z:0^lstm_10/while/NoOp*
T0*'
_output_shapes
:         Z­
lstm_10/while/NoOpNoOp2^lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp1^lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp3^lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_10_while_identitylstm_10/while/Identity:output:0"=
lstm_10_while_identity_1!lstm_10/while/Identity_1:output:0"=
lstm_10_while_identity_2!lstm_10/while/Identity_2:output:0"=
lstm_10_while_identity_3!lstm_10/while/Identity_3:output:0"=
lstm_10_while_identity_4!lstm_10/while/Identity_4:output:0"=
lstm_10_while_identity_5!lstm_10/while/Identity_5:output:0"P
%lstm_10_while_lstm_10_strided_slice_1'lstm_10_while_lstm_10_strided_slice_1_0"z
:lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource<lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0"|
;lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource=lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0"x
9lstm_10_while_lstm_cell_10_matmul_readvariableop_resource;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0"╚
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2f
1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp2d
0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp2h
2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
╬"
▄
while_body_197735
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_9_197759_0:	У-
while_lstm_cell_9_197761_0:	ZУ)
while_lstm_cell_9_197763_0:	У
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_9_197759:	У+
while_lstm_cell_9_197761:	ZУ'
while_lstm_cell_9_197763:	Уѕб)while/lstm_cell_9/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0«
)while/lstm_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_9_197759_0while_lstm_cell_9_197761_0while_lstm_cell_9_197763_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         Z:         Z:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_197676█
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_9/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ј
while/Identity_4Identity2while/lstm_cell_9/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         ZЈ
while/Identity_5Identity2while/lstm_cell_9/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         Zx

while/NoOpNoOp*^while/lstm_cell_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_9_197759while_lstm_cell_9_197759_0"6
while_lstm_cell_9_197761while_lstm_cell_9_197761_0"6
while_lstm_cell_9_197763while_lstm_cell_9_197763_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2V
)while/lstm_cell_9/StatefulPartitionedCall)while/lstm_cell_9/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
ѓ8
ђ
B__inference_lstm_9_layer_call_and_return_conditional_losses_197613

inputs%
lstm_cell_9_197531:	У%
lstm_cell_9_197533:	ZУ!
lstm_cell_9_197535:	У
identityѕб#lstm_cell_9/StatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zs
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         ZR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Zw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         Zc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask­
#lstm_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_9_197531lstm_cell_9_197533lstm_cell_9_197535*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         Z:         Z:         Z*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_197530n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┤
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_9_197531lstm_cell_9_197533lstm_cell_9_197535*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         Z:         Z: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_197544*
condR
while_cond_197543*K
output_shapes:
8: : : : :         Z:         Z: : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  Z*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         Z*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  Z[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Zt
NoOpNoOp$^lstm_cell_9/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_9/StatefulPartitionedCall#lstm_cell_9/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Ъ
F
*__inference_dropout_4_layer_call_fn_200920

inputs
identity░
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_198481`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         Z"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         Z:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs
жB
л

lstm_10_while_body_199576,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3+
'lstm_10_while_lstm_10_strided_slice_1_0g
clstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0:	ZУP
=lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0:	ZУK
<lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0:	У
lstm_10_while_identity
lstm_10_while_identity_1
lstm_10_while_identity_2
lstm_10_while_identity_3
lstm_10_while_identity_4
lstm_10_while_identity_5)
%lstm_10_while_lstm_10_strided_slice_1e
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorL
9lstm_10_while_lstm_cell_10_matmul_readvariableop_resource:	ZУN
;lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource:	ZУI
:lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource:	Уѕб1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpб0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpб2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpљ
?lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Z   ╬
1lstm_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0lstm_10_while_placeholderHlstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         Z*
element_dtype0Г
0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0м
!lstm_10/while/lstm_cell_10/MatMulMatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         У▒
2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp=lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	ZУ*
dtype0╣
#lstm_10/while/lstm_cell_10/MatMul_1MatMullstm_10_while_placeholder_2:lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         УХ
lstm_10/while/lstm_cell_10/addAddV2+lstm_10/while/lstm_cell_10/MatMul:product:0-lstm_10/while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         УФ
1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp<lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:У*
dtype0┐
"lstm_10/while/lstm_cell_10/BiasAddBiasAdd"lstm_10/while/lstm_cell_10/add:z:09lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Уl
*lstm_10/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Є
 lstm_10/while/lstm_cell_10/splitSplit3lstm_10/while/lstm_cell_10/split/split_dim:output:0+lstm_10/while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         Z:         Z:         Z:         Z*
	num_splitі
"lstm_10/while/lstm_cell_10/SigmoidSigmoid)lstm_10/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         Zї
$lstm_10/while/lstm_cell_10/Sigmoid_1Sigmoid)lstm_10/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         Zъ
lstm_10/while/lstm_cell_10/mulMul(lstm_10/while/lstm_cell_10/Sigmoid_1:y:0lstm_10_while_placeholder_3*
T0*'
_output_shapes
:         Zё
lstm_10/while/lstm_cell_10/ReluRelu)lstm_10/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         Z░
 lstm_10/while/lstm_cell_10/mul_1Mul&lstm_10/while/lstm_cell_10/Sigmoid:y:0-lstm_10/while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         ZЦ
 lstm_10/while/lstm_cell_10/add_1AddV2"lstm_10/while/lstm_cell_10/mul:z:0$lstm_10/while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         Zї
$lstm_10/while/lstm_cell_10/Sigmoid_2Sigmoid)lstm_10/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         ZЂ
!lstm_10/while/lstm_cell_10/Relu_1Relu$lstm_10/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         Z┤
 lstm_10/while/lstm_cell_10/mul_2Mul(lstm_10/while/lstm_cell_10/Sigmoid_2:y:0/lstm_10/while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         Zz
8lstm_10/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Ї
2lstm_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_10_while_placeholder_1Alstm_10/while/TensorArrayV2Write/TensorListSetItem/index:output:0$lstm_10/while/lstm_cell_10/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмU
lstm_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_10/while/addAddV2lstm_10_while_placeholderlstm_10/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Є
lstm_10/while/add_1AddV2(lstm_10_while_lstm_10_while_loop_counterlstm_10/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_10/while/IdentityIdentitylstm_10/while/add_1:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: і
lstm_10/while/Identity_1Identity.lstm_10_while_lstm_10_while_maximum_iterations^lstm_10/while/NoOp*
T0*
_output_shapes
: q
lstm_10/while/Identity_2Identitylstm_10/while/add:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: ъ
lstm_10/while/Identity_3IdentityBlstm_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_10/while/NoOp*
T0*
_output_shapes
: Љ
lstm_10/while/Identity_4Identity$lstm_10/while/lstm_cell_10/mul_2:z:0^lstm_10/while/NoOp*
T0*'
_output_shapes
:         ZЉ
lstm_10/while/Identity_5Identity$lstm_10/while/lstm_cell_10/add_1:z:0^lstm_10/while/NoOp*
T0*'
_output_shapes
:         Z­
lstm_10/while/NoOpNoOp2^lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp1^lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp3^lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_10_while_identitylstm_10/while/Identity:output:0"=
lstm_10_while_identity_1!lstm_10/while/Identity_1:output:0"=
lstm_10_while_identity_2!lstm_10/while/Identity_2:output:0"=
lstm_10_while_identity_3!lstm_10/while/Identity_3:output:0"=
lstm_10_while_identity_4!lstm_10/while/Identity_4:output:0"=
lstm_10_while_identity_5!lstm_10/while/Identity_5:output:0"P
%lstm_10_while_lstm_10_strided_slice_1'lstm_10_while_lstm_10_strided_slice_1_0"z
:lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource<lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0"|
;lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource=lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0"x
9lstm_10_while_lstm_cell_10_matmul_readvariableop_resource;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0"╚
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         Z:         Z: : : : : 2f
1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp2d
0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp2h
2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
: 
х
├
while_cond_199920
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_199920___redundant_placeholder04
0while_while_cond_199920___redundant_placeholder14
0while_while_cond_199920___redundant_placeholder24
0while_while_cond_199920___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
щ	
¤
lstm_9_while_cond_199144*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3,
(lstm_9_while_less_lstm_9_strided_slice_1B
>lstm_9_while_lstm_9_while_cond_199144___redundant_placeholder0B
>lstm_9_while_lstm_9_while_cond_199144___redundant_placeholder1B
>lstm_9_while_lstm_9_while_cond_199144___redundant_placeholder2B
>lstm_9_while_lstm_9_while_cond_199144___redundant_placeholder3
lstm_9_while_identity
~
lstm_9/while/LessLesslstm_9_while_placeholder(lstm_9_while_less_lstm_9_strided_slice_1*
T0*
_output_shapes
: Y
lstm_9/while/IdentityIdentitylstm_9/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_9_while_identitylstm_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
х
├
while_cond_198624
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_198624___redundant_placeholder04
0while_while_cond_198624___redundant_placeholder14
0while_while_cond_198624___redundant_placeholder24
0while_while_cond_198624___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         Z:         Z: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         Z:-)
'
_output_shapes
:         Z:

_output_shapes
: :

_output_shapes
:
ы
c
*__inference_dropout_4_layer_call_fn_200925

inputs
identityѕбStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_198549o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Z`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         Z22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         Z
 
_user_specified_nameinputs"є
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*И
serving_defaultц
I
lstm_9_input9
serving_default_lstm_9_input:0         ;
dense_40
StatefulPartitionedCall:0         tensorflow/serving/predict:№І
У
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
┌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
┌
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
╝
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&_random_generator"
_tf_keras_layer
╗
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias"
_tf_keras_layer
X
/0
01
12
23
34
45
-6
.7"
trackable_list_wrapper
X
/0
01
12
23
34
45
-6
.7"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
ж
:trace_0
;trace_1
<trace_2
=trace_32■
-__inference_sequential_4_layer_call_fn_198519
-__inference_sequential_4_layer_call_fn_199065
-__inference_sequential_4_layer_call_fn_199086
-__inference_sequential_4_layer_call_fn_198971┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z:trace_0z;trace_1z<trace_2z=trace_3
Н
>trace_0
?trace_1
@trace_2
Atrace_32Ж
H__inference_sequential_4_layer_call_and_return_conditional_losses_199377
H__inference_sequential_4_layer_call_and_return_conditional_losses_199675
H__inference_sequential_4_layer_call_and_return_conditional_losses_198995
H__inference_sequential_4_layer_call_and_return_conditional_losses_199019┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z>trace_0z?trace_1z@trace_2zAtrace_3
ЛB╬
!__inference__wrapped_model_197463lstm_9_input"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ю
B
_variables
C_iterations
D_learning_rate
E_index_dict
F
_momentums
G_velocities
H_update_step_xla"
experimentalOptimizer
,
Iserving_default"
signature_map
5
/0
01
12"
trackable_list_wrapper
5
/0
01
12"
trackable_list_wrapper
 "
trackable_list_wrapper
╣

Jstates
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Т
Ptrace_0
Qtrace_1
Rtrace_2
Strace_32ч
'__inference_lstm_9_layer_call_fn_199686
'__inference_lstm_9_layer_call_fn_199697
'__inference_lstm_9_layer_call_fn_199708
'__inference_lstm_9_layer_call_fn_199719н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zPtrace_0zQtrace_1zRtrace_2zStrace_3
м
Ttrace_0
Utrace_1
Vtrace_2
Wtrace_32у
B__inference_lstm_9_layer_call_and_return_conditional_losses_199862
B__inference_lstm_9_layer_call_and_return_conditional_losses_200005
B__inference_lstm_9_layer_call_and_return_conditional_losses_200148
B__inference_lstm_9_layer_call_and_return_conditional_losses_200291н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zTtrace_0zUtrace_1zVtrace_2zWtrace_3
"
_generic_user_object
Э
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^_random_generator
_
state_size

/kernel
0recurrent_kernel
1bias"
_tf_keras_layer
 "
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
 "
trackable_list_wrapper
╣

`states
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ж
ftrace_0
gtrace_1
htrace_2
itrace_32 
(__inference_lstm_10_layer_call_fn_200302
(__inference_lstm_10_layer_call_fn_200313
(__inference_lstm_10_layer_call_fn_200324
(__inference_lstm_10_layer_call_fn_200335н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zftrace_0zgtrace_1zhtrace_2zitrace_3
о
jtrace_0
ktrace_1
ltrace_2
mtrace_32в
C__inference_lstm_10_layer_call_and_return_conditional_losses_200480
C__inference_lstm_10_layer_call_and_return_conditional_losses_200625
C__inference_lstm_10_layer_call_and_return_conditional_losses_200770
C__inference_lstm_10_layer_call_and_return_conditional_losses_200915н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zjtrace_0zktrace_1zltrace_2zmtrace_3
"
_generic_user_object
Э
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
t_random_generator
u
state_size

2kernel
3recurrent_kernel
4bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
┼
{trace_0
|trace_12ј
*__inference_dropout_4_layer_call_fn_200920
*__inference_dropout_4_layer_call_fn_200925│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z{trace_0z|trace_1
ч
}trace_0
~trace_12─
E__inference_dropout_4_layer_call_and_return_conditional_losses_200930
E__inference_dropout_4_layer_call_and_return_conditional_losses_200942│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z}trace_0z~trace_1
"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
▒
non_trainable_variables
ђlayers
Ђmetrics
 ѓlayer_regularization_losses
Ѓlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
Ь
ёtrace_02¤
(__inference_dense_4_layer_call_fn_200951б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zёtrace_0
Ѕ
Ёtrace_02Ж
C__inference_dense_4_layer_call_and_return_conditional_losses_200961б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЁtrace_0
 :Z2dense_4/kernel
:2dense_4/bias
,:*	У2lstm_9/lstm_cell_9/kernel
6:4	ZУ2#lstm_9/lstm_cell_9/recurrent_kernel
&:$У2lstm_9/lstm_cell_9/bias
.:,	ZУ2lstm_10/lstm_cell_10/kernel
8:6	ZУ2%lstm_10/lstm_cell_10/recurrent_kernel
(:&У2lstm_10/lstm_cell_10/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
0
є0
Є1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ёBЂ
-__inference_sequential_4_layer_call_fn_198519lstm_9_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
-__inference_sequential_4_layer_call_fn_199065inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
-__inference_sequential_4_layer_call_fn_199086inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ёBЂ
-__inference_sequential_4_layer_call_fn_198971lstm_9_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
H__inference_sequential_4_layer_call_and_return_conditional_losses_199377inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
H__inference_sequential_4_layer_call_and_return_conditional_losses_199675inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЪBю
H__inference_sequential_4_layer_call_and_return_conditional_losses_198995lstm_9_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЪBю
H__inference_sequential_4_layer_call_and_return_conditional_losses_199019lstm_9_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
«
C0
ѕ1
Ѕ2
і3
І4
ї5
Ї6
ј7
Ј8
љ9
Љ10
њ11
Њ12
ћ13
Ћ14
ќ15
Ќ16"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
`
ѕ0
і1
ї2
ј3
љ4
њ5
ћ6
ќ7"
trackable_list_wrapper
`
Ѕ0
І1
Ї2
Ј3
Љ4
Њ5
Ћ6
Ќ7"
trackable_list_wrapper
┐2╝╣
«▓ф
FullArgSpec2
args*џ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
лB═
$__inference_signature_wrapper_199044lstm_9_input"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЈBї
'__inference_lstm_9_layer_call_fn_199686inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЈBї
'__inference_lstm_9_layer_call_fn_199697inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЇBі
'__inference_lstm_9_layer_call_fn_199708inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЇBі
'__inference_lstm_9_layer_call_fn_199719inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
фBД
B__inference_lstm_9_layer_call_and_return_conditional_losses_199862inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
фBД
B__inference_lstm_9_layer_call_and_return_conditional_losses_200005inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
еBЦ
B__inference_lstm_9_layer_call_and_return_conditional_losses_200148inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
еBЦ
B__inference_lstm_9_layer_call_and_return_conditional_losses_200291inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
5
/0
01
12"
trackable_list_wrapper
5
/0
01
12"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ўnon_trainable_variables
Ўlayers
џmetrics
 Џlayer_regularization_losses
юlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
О
Юtrace_0
ъtrace_12ю
,__inference_lstm_cell_9_layer_call_fn_200978
,__inference_lstm_cell_9_layer_call_fn_200995й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЮtrace_0zъtrace_1
Ї
Ъtrace_0
аtrace_12м
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_201027
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_201059й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЪtrace_0zаtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
љBЇ
(__inference_lstm_10_layer_call_fn_200302inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
љBЇ
(__inference_lstm_10_layer_call_fn_200313inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
јBІ
(__inference_lstm_10_layer_call_fn_200324inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
јBІ
(__inference_lstm_10_layer_call_fn_200335inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ФBе
C__inference_lstm_10_layer_call_and_return_conditional_losses_200480inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ФBе
C__inference_lstm_10_layer_call_and_return_conditional_losses_200625inputs_0"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЕBд
C__inference_lstm_10_layer_call_and_return_conditional_losses_200770inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЕBд
C__inference_lstm_10_layer_call_and_return_conditional_losses_200915inputs"н
╦▓К
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
5
20
31
42"
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Аnon_trainable_variables
бlayers
Бmetrics
 цlayer_regularization_losses
Цlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
┘
дtrace_0
Дtrace_12ъ
-__inference_lstm_cell_10_layer_call_fn_201076
-__inference_lstm_cell_10_layer_call_fn_201093й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zдtrace_0zДtrace_1
Ј
еtrace_0
Еtrace_12н
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_201125
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_201157й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zеtrace_0zЕtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№BВ
*__inference_dropout_4_layer_call_fn_200920inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№BВ
*__inference_dropout_4_layer_call_fn_200925inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
іBЄ
E__inference_dropout_4_layer_call_and_return_conditional_losses_200930inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
іBЄ
E__inference_dropout_4_layer_call_and_return_conditional_losses_200942inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▄B┘
(__inference_dense_4_layer_call_fn_200951inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
C__inference_dense_4_layer_call_and_return_conditional_losses_200961inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
R
ф	variables
Ф	keras_api

гtotal

Гcount"
_tf_keras_metric
c
«	variables
»	keras_api

░total

▒count
▓
_fn_kwargs"
_tf_keras_metric
1:/	У2 Adam/m/lstm_9/lstm_cell_9/kernel
1:/	У2 Adam/v/lstm_9/lstm_cell_9/kernel
;:9	ZУ2*Adam/m/lstm_9/lstm_cell_9/recurrent_kernel
;:9	ZУ2*Adam/v/lstm_9/lstm_cell_9/recurrent_kernel
+:)У2Adam/m/lstm_9/lstm_cell_9/bias
+:)У2Adam/v/lstm_9/lstm_cell_9/bias
3:1	ZУ2"Adam/m/lstm_10/lstm_cell_10/kernel
3:1	ZУ2"Adam/v/lstm_10/lstm_cell_10/kernel
=:;	ZУ2,Adam/m/lstm_10/lstm_cell_10/recurrent_kernel
=:;	ZУ2,Adam/v/lstm_10/lstm_cell_10/recurrent_kernel
-:+У2 Adam/m/lstm_10/lstm_cell_10/bias
-:+У2 Adam/v/lstm_10/lstm_cell_10/bias
%:#Z2Adam/m/dense_4/kernel
%:#Z2Adam/v/dense_4/kernel
:2Adam/m/dense_4/bias
:2Adam/v/dense_4/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЈBї
,__inference_lstm_cell_9_layer_call_fn_200978inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЈBї
,__inference_lstm_cell_9_layer_call_fn_200995inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
фBД
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_201027inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
фBД
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_201059inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
љBЇ
-__inference_lstm_cell_10_layer_call_fn_201076inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
љBЇ
-__inference_lstm_cell_10_layer_call_fn_201093inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ФBе
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_201125inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ФBе
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_201157inputsstates_0states_1"й
┤▓░
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
г0
Г1"
trackable_list_wrapper
.
ф	variables"
_generic_user_object
:  (2total
:  (2count
0
░0
▒1"
trackable_list_wrapper
.
«	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperЮ
!__inference__wrapped_model_197463x/01234-.9б6
/б,
*і'
lstm_9_input         
ф "1ф.
,
dense_4!і
dense_4         ф
C__inference_dense_4_layer_call_and_return_conditional_losses_200961c-./б,
%б"
 і
inputs         Z
ф ",б)
"і
tensor_0         
џ ё
(__inference_dense_4_layer_call_fn_200951X-./б,
%б"
 і
inputs         Z
ф "!і
unknown         г
E__inference_dropout_4_layer_call_and_return_conditional_losses_200930c3б0
)б&
 і
inputs         Z
p 
ф ",б)
"і
tensor_0         Z
џ г
E__inference_dropout_4_layer_call_and_return_conditional_losses_200942c3б0
)б&
 і
inputs         Z
p
ф ",б)
"і
tensor_0         Z
џ є
*__inference_dropout_4_layer_call_fn_200920X3б0
)б&
 і
inputs         Z
p 
ф "!і
unknown         Zє
*__inference_dropout_4_layer_call_fn_200925X3б0
)б&
 і
inputs         Z
p
ф "!і
unknown         Z╠
C__inference_lstm_10_layer_call_and_return_conditional_losses_200480ё234OбL
EбB
4џ1
/і,
inputs_0                  Z

 
p 

 
ф ",б)
"і
tensor_0         Z
џ ╠
C__inference_lstm_10_layer_call_and_return_conditional_losses_200625ё234OбL
EбB
4џ1
/і,
inputs_0                  Z

 
p

 
ф ",б)
"і
tensor_0         Z
џ ╗
C__inference_lstm_10_layer_call_and_return_conditional_losses_200770t234?б<
5б2
$і!
inputs         Z

 
p 

 
ф ",б)
"і
tensor_0         Z
џ ╗
C__inference_lstm_10_layer_call_and_return_conditional_losses_200915t234?б<
5б2
$і!
inputs         Z

 
p

 
ф ",б)
"і
tensor_0         Z
џ Ц
(__inference_lstm_10_layer_call_fn_200302y234OбL
EбB
4џ1
/і,
inputs_0                  Z

 
p 

 
ф "!і
unknown         ZЦ
(__inference_lstm_10_layer_call_fn_200313y234OбL
EбB
4џ1
/і,
inputs_0                  Z

 
p

 
ф "!і
unknown         ZЋ
(__inference_lstm_10_layer_call_fn_200324i234?б<
5б2
$і!
inputs         Z

 
p 

 
ф "!і
unknown         ZЋ
(__inference_lstm_10_layer_call_fn_200335i234?б<
5б2
$і!
inputs         Z

 
p

 
ф "!і
unknown         Zп
B__inference_lstm_9_layer_call_and_return_conditional_losses_199862Љ/01OбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф "9б6
/і,
tensor_0                  Z
џ п
B__inference_lstm_9_layer_call_and_return_conditional_losses_200005Љ/01OбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф "9б6
/і,
tensor_0                  Z
џ Й
B__inference_lstm_9_layer_call_and_return_conditional_losses_200148x/01?б<
5б2
$і!
inputs         

 
p 

 
ф "0б-
&і#
tensor_0         Z
џ Й
B__inference_lstm_9_layer_call_and_return_conditional_losses_200291x/01?б<
5б2
$і!
inputs         

 
p

 
ф "0б-
&і#
tensor_0         Z
џ ▓
'__inference_lstm_9_layer_call_fn_199686є/01OбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф ".і+
unknown                  Z▓
'__inference_lstm_9_layer_call_fn_199697є/01OбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф ".і+
unknown                  Zў
'__inference_lstm_9_layer_call_fn_199708m/01?б<
5б2
$і!
inputs         

 
p 

 
ф "%і"
unknown         Zў
'__inference_lstm_9_layer_call_fn_199719m/01?б<
5б2
$і!
inputs         

 
p

 
ф "%і"
unknown         Zр
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_201125ћ234ђб}
vбs
 і
inputs         Z
KбH
"і
states_0         Z
"і
states_1         Z
p 
ф "ЅбЁ
~б{
$і!

tensor_0_0         Z
SџP
&і#
tensor_0_1_0         Z
&і#
tensor_0_1_1         Z
џ р
H__inference_lstm_cell_10_layer_call_and_return_conditional_losses_201157ћ234ђб}
vбs
 і
inputs         Z
KбH
"і
states_0         Z
"і
states_1         Z
p
ф "ЅбЁ
~б{
$і!

tensor_0_0         Z
SџP
&і#
tensor_0_1_0         Z
&і#
tensor_0_1_1         Z
џ ┤
-__inference_lstm_cell_10_layer_call_fn_201076ѓ234ђб}
vбs
 і
inputs         Z
KбH
"і
states_0         Z
"і
states_1         Z
p 
ф "xбu
"і
tensor_0         Z
OџL
$і!

tensor_1_0         Z
$і!

tensor_1_1         Z┤
-__inference_lstm_cell_10_layer_call_fn_201093ѓ234ђб}
vбs
 і
inputs         Z
KбH
"і
states_0         Z
"і
states_1         Z
p
ф "xбu
"і
tensor_0         Z
OџL
$і!

tensor_1_0         Z
$і!

tensor_1_1         ZЯ
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_201027ћ/01ђб}
vбs
 і
inputs         
KбH
"і
states_0         Z
"і
states_1         Z
p 
ф "ЅбЁ
~б{
$і!

tensor_0_0         Z
SџP
&і#
tensor_0_1_0         Z
&і#
tensor_0_1_1         Z
џ Я
G__inference_lstm_cell_9_layer_call_and_return_conditional_losses_201059ћ/01ђб}
vбs
 і
inputs         
KбH
"і
states_0         Z
"і
states_1         Z
p
ф "ЅбЁ
~б{
$і!

tensor_0_0         Z
SџP
&і#
tensor_0_1_0         Z
&і#
tensor_0_1_1         Z
џ │
,__inference_lstm_cell_9_layer_call_fn_200978ѓ/01ђб}
vбs
 і
inputs         
KбH
"і
states_0         Z
"і
states_1         Z
p 
ф "xбu
"і
tensor_0         Z
OџL
$і!

tensor_1_0         Z
$і!

tensor_1_1         Z│
,__inference_lstm_cell_9_layer_call_fn_200995ѓ/01ђб}
vбs
 і
inputs         
KбH
"і
states_0         Z
"і
states_1         Z
p
ф "xбu
"і
tensor_0         Z
OџL
$і!

tensor_1_0         Z
$і!

tensor_1_1         ZК
H__inference_sequential_4_layer_call_and_return_conditional_losses_198995{/01234-.Aб>
7б4
*і'
lstm_9_input         
p 

 
ф ",б)
"і
tensor_0         
џ К
H__inference_sequential_4_layer_call_and_return_conditional_losses_199019{/01234-.Aб>
7б4
*і'
lstm_9_input         
p

 
ф ",б)
"і
tensor_0         
џ ┴
H__inference_sequential_4_layer_call_and_return_conditional_losses_199377u/01234-.;б8
1б.
$і!
inputs         
p 

 
ф ",б)
"і
tensor_0         
џ ┴
H__inference_sequential_4_layer_call_and_return_conditional_losses_199675u/01234-.;б8
1б.
$і!
inputs         
p

 
ф ",б)
"і
tensor_0         
џ А
-__inference_sequential_4_layer_call_fn_198519p/01234-.Aб>
7б4
*і'
lstm_9_input         
p 

 
ф "!і
unknown         А
-__inference_sequential_4_layer_call_fn_198971p/01234-.Aб>
7б4
*і'
lstm_9_input         
p

 
ф "!і
unknown         Џ
-__inference_sequential_4_layer_call_fn_199065j/01234-.;б8
1б.
$і!
inputs         
p 

 
ф "!і
unknown         Џ
-__inference_sequential_4_layer_call_fn_199086j/01234-.;б8
1б.
$і!
inputs         
p

 
ф "!і
unknown         ▒
$__inference_signature_wrapper_199044ѕ/01234-.IбF
б 
?ф<
:
lstm_9_input*і'
lstm_9_input         "1ф.
,
dense_4!і
dense_4         