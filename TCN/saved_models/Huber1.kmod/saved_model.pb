ΞΠ%
Ρ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
Ύ
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878ωΠ 
|
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_34/kernel
u
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel* 
_output_shapes
:
*
dtype0
s
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_34/bias
l
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
_output_shapes	
:*
dtype0
|
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_35/kernel
u
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel* 
_output_shapes
:
*
dtype0
s
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_35/bias
l
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
_output_shapes	
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
―
'tcn_17/residual_block_0/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'tcn_17/residual_block_0/conv1D_0/kernel
¨
;tcn_17/residual_block_0/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp'tcn_17/residual_block_0/conv1D_0/kernel*#
_output_shapes
:@*
dtype0
£
%tcn_17/residual_block_0/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%tcn_17/residual_block_0/conv1D_0/bias

9tcn_17/residual_block_0/conv1D_0/bias/Read/ReadVariableOpReadVariableOp%tcn_17/residual_block_0/conv1D_0/bias*
_output_shapes	
:*
dtype0
°
'tcn_17/residual_block_0/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'tcn_17/residual_block_0/conv1D_1/kernel
©
;tcn_17/residual_block_0/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp'tcn_17/residual_block_0/conv1D_1/kernel*$
_output_shapes
:@*
dtype0
£
%tcn_17/residual_block_0/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%tcn_17/residual_block_0/conv1D_1/bias

9tcn_17/residual_block_0/conv1D_1/bias/Read/ReadVariableOpReadVariableOp%tcn_17/residual_block_0/conv1D_1/bias*
_output_shapes	
:*
dtype0
½
.tcn_17/residual_block_0/matching_conv1D/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.tcn_17/residual_block_0/matching_conv1D/kernel
Ά
Btcn_17/residual_block_0/matching_conv1D/kernel/Read/ReadVariableOpReadVariableOp.tcn_17/residual_block_0/matching_conv1D/kernel*#
_output_shapes
:*
dtype0
±
,tcn_17/residual_block_0/matching_conv1D/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,tcn_17/residual_block_0/matching_conv1D/bias
ͺ
@tcn_17/residual_block_0/matching_conv1D/bias/Read/ReadVariableOpReadVariableOp,tcn_17/residual_block_0/matching_conv1D/bias*
_output_shapes	
:*
dtype0
°
'tcn_17/residual_block_1/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'tcn_17/residual_block_1/conv1D_0/kernel
©
;tcn_17/residual_block_1/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp'tcn_17/residual_block_1/conv1D_0/kernel*$
_output_shapes
:@*
dtype0
£
%tcn_17/residual_block_1/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%tcn_17/residual_block_1/conv1D_0/bias

9tcn_17/residual_block_1/conv1D_0/bias/Read/ReadVariableOpReadVariableOp%tcn_17/residual_block_1/conv1D_0/bias*
_output_shapes	
:*
dtype0
°
'tcn_17/residual_block_1/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'tcn_17/residual_block_1/conv1D_1/kernel
©
;tcn_17/residual_block_1/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp'tcn_17/residual_block_1/conv1D_1/kernel*$
_output_shapes
:@*
dtype0
£
%tcn_17/residual_block_1/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%tcn_17/residual_block_1/conv1D_1/bias

9tcn_17/residual_block_1/conv1D_1/bias/Read/ReadVariableOpReadVariableOp%tcn_17/residual_block_1/conv1D_1/bias*
_output_shapes	
:*
dtype0
°
'tcn_17/residual_block_2/conv1D_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'tcn_17/residual_block_2/conv1D_0/kernel
©
;tcn_17/residual_block_2/conv1D_0/kernel/Read/ReadVariableOpReadVariableOp'tcn_17/residual_block_2/conv1D_0/kernel*$
_output_shapes
:@*
dtype0
£
%tcn_17/residual_block_2/conv1D_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%tcn_17/residual_block_2/conv1D_0/bias

9tcn_17/residual_block_2/conv1D_0/bias/Read/ReadVariableOpReadVariableOp%tcn_17/residual_block_2/conv1D_0/bias*
_output_shapes	
:*
dtype0
°
'tcn_17/residual_block_2/conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'tcn_17/residual_block_2/conv1D_1/kernel
©
;tcn_17/residual_block_2/conv1D_1/kernel/Read/ReadVariableOpReadVariableOp'tcn_17/residual_block_2/conv1D_1/kernel*$
_output_shapes
:@*
dtype0
£
%tcn_17/residual_block_2/conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%tcn_17/residual_block_2/conv1D_1/bias

9tcn_17/residual_block_2/conv1D_1/bias/Read/ReadVariableOpReadVariableOp%tcn_17/residual_block_2/conv1D_1/bias*
_output_shapes	
:*
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

Adam/dense_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_34/kernel/m

*Adam/dense_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_34/bias/m
z
(Adam/dense_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_35/kernel/m

*Adam/dense_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_35/bias/m
z
(Adam/dense_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/m*
_output_shapes	
:*
dtype0
½
.Adam/tcn_17/residual_block_0/conv1D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/tcn_17/residual_block_0/conv1D_0/kernel/m
Ά
BAdam/tcn_17/residual_block_0/conv1D_0/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/tcn_17/residual_block_0/conv1D_0/kernel/m*#
_output_shapes
:@*
dtype0
±
,Adam/tcn_17/residual_block_0/conv1D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/tcn_17/residual_block_0/conv1D_0/bias/m
ͺ
@Adam/tcn_17/residual_block_0/conv1D_0/bias/m/Read/ReadVariableOpReadVariableOp,Adam/tcn_17/residual_block_0/conv1D_0/bias/m*
_output_shapes	
:*
dtype0
Ύ
.Adam/tcn_17/residual_block_0/conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/tcn_17/residual_block_0/conv1D_1/kernel/m
·
BAdam/tcn_17/residual_block_0/conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/tcn_17/residual_block_0/conv1D_1/kernel/m*$
_output_shapes
:@*
dtype0
±
,Adam/tcn_17/residual_block_0/conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/tcn_17/residual_block_0/conv1D_1/bias/m
ͺ
@Adam/tcn_17/residual_block_0/conv1D_1/bias/m/Read/ReadVariableOpReadVariableOp,Adam/tcn_17/residual_block_0/conv1D_1/bias/m*
_output_shapes	
:*
dtype0
Λ
5Adam/tcn_17/residual_block_0/matching_conv1D/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/tcn_17/residual_block_0/matching_conv1D/kernel/m
Δ
IAdam/tcn_17/residual_block_0/matching_conv1D/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/tcn_17/residual_block_0/matching_conv1D/kernel/m*#
_output_shapes
:*
dtype0
Ώ
3Adam/tcn_17/residual_block_0/matching_conv1D/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/tcn_17/residual_block_0/matching_conv1D/bias/m
Έ
GAdam/tcn_17/residual_block_0/matching_conv1D/bias/m/Read/ReadVariableOpReadVariableOp3Adam/tcn_17/residual_block_0/matching_conv1D/bias/m*
_output_shapes	
:*
dtype0
Ύ
.Adam/tcn_17/residual_block_1/conv1D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/tcn_17/residual_block_1/conv1D_0/kernel/m
·
BAdam/tcn_17/residual_block_1/conv1D_0/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/tcn_17/residual_block_1/conv1D_0/kernel/m*$
_output_shapes
:@*
dtype0
±
,Adam/tcn_17/residual_block_1/conv1D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/tcn_17/residual_block_1/conv1D_0/bias/m
ͺ
@Adam/tcn_17/residual_block_1/conv1D_0/bias/m/Read/ReadVariableOpReadVariableOp,Adam/tcn_17/residual_block_1/conv1D_0/bias/m*
_output_shapes	
:*
dtype0
Ύ
.Adam/tcn_17/residual_block_1/conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/tcn_17/residual_block_1/conv1D_1/kernel/m
·
BAdam/tcn_17/residual_block_1/conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/tcn_17/residual_block_1/conv1D_1/kernel/m*$
_output_shapes
:@*
dtype0
±
,Adam/tcn_17/residual_block_1/conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/tcn_17/residual_block_1/conv1D_1/bias/m
ͺ
@Adam/tcn_17/residual_block_1/conv1D_1/bias/m/Read/ReadVariableOpReadVariableOp,Adam/tcn_17/residual_block_1/conv1D_1/bias/m*
_output_shapes	
:*
dtype0
Ύ
.Adam/tcn_17/residual_block_2/conv1D_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/tcn_17/residual_block_2/conv1D_0/kernel/m
·
BAdam/tcn_17/residual_block_2/conv1D_0/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/tcn_17/residual_block_2/conv1D_0/kernel/m*$
_output_shapes
:@*
dtype0
±
,Adam/tcn_17/residual_block_2/conv1D_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/tcn_17/residual_block_2/conv1D_0/bias/m
ͺ
@Adam/tcn_17/residual_block_2/conv1D_0/bias/m/Read/ReadVariableOpReadVariableOp,Adam/tcn_17/residual_block_2/conv1D_0/bias/m*
_output_shapes	
:*
dtype0
Ύ
.Adam/tcn_17/residual_block_2/conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/tcn_17/residual_block_2/conv1D_1/kernel/m
·
BAdam/tcn_17/residual_block_2/conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/tcn_17/residual_block_2/conv1D_1/kernel/m*$
_output_shapes
:@*
dtype0
±
,Adam/tcn_17/residual_block_2/conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/tcn_17/residual_block_2/conv1D_1/bias/m
ͺ
@Adam/tcn_17/residual_block_2/conv1D_1/bias/m/Read/ReadVariableOpReadVariableOp,Adam/tcn_17/residual_block_2/conv1D_1/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_34/kernel/v

*Adam/dense_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_34/bias/v
z
(Adam/dense_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_35/kernel/v

*Adam/dense_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_35/bias/v
z
(Adam/dense_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/v*
_output_shapes	
:*
dtype0
½
.Adam/tcn_17/residual_block_0/conv1D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/tcn_17/residual_block_0/conv1D_0/kernel/v
Ά
BAdam/tcn_17/residual_block_0/conv1D_0/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/tcn_17/residual_block_0/conv1D_0/kernel/v*#
_output_shapes
:@*
dtype0
±
,Adam/tcn_17/residual_block_0/conv1D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/tcn_17/residual_block_0/conv1D_0/bias/v
ͺ
@Adam/tcn_17/residual_block_0/conv1D_0/bias/v/Read/ReadVariableOpReadVariableOp,Adam/tcn_17/residual_block_0/conv1D_0/bias/v*
_output_shapes	
:*
dtype0
Ύ
.Adam/tcn_17/residual_block_0/conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/tcn_17/residual_block_0/conv1D_1/kernel/v
·
BAdam/tcn_17/residual_block_0/conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/tcn_17/residual_block_0/conv1D_1/kernel/v*$
_output_shapes
:@*
dtype0
±
,Adam/tcn_17/residual_block_0/conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/tcn_17/residual_block_0/conv1D_1/bias/v
ͺ
@Adam/tcn_17/residual_block_0/conv1D_1/bias/v/Read/ReadVariableOpReadVariableOp,Adam/tcn_17/residual_block_0/conv1D_1/bias/v*
_output_shapes	
:*
dtype0
Λ
5Adam/tcn_17/residual_block_0/matching_conv1D/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/tcn_17/residual_block_0/matching_conv1D/kernel/v
Δ
IAdam/tcn_17/residual_block_0/matching_conv1D/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/tcn_17/residual_block_0/matching_conv1D/kernel/v*#
_output_shapes
:*
dtype0
Ώ
3Adam/tcn_17/residual_block_0/matching_conv1D/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adam/tcn_17/residual_block_0/matching_conv1D/bias/v
Έ
GAdam/tcn_17/residual_block_0/matching_conv1D/bias/v/Read/ReadVariableOpReadVariableOp3Adam/tcn_17/residual_block_0/matching_conv1D/bias/v*
_output_shapes	
:*
dtype0
Ύ
.Adam/tcn_17/residual_block_1/conv1D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/tcn_17/residual_block_1/conv1D_0/kernel/v
·
BAdam/tcn_17/residual_block_1/conv1D_0/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/tcn_17/residual_block_1/conv1D_0/kernel/v*$
_output_shapes
:@*
dtype0
±
,Adam/tcn_17/residual_block_1/conv1D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/tcn_17/residual_block_1/conv1D_0/bias/v
ͺ
@Adam/tcn_17/residual_block_1/conv1D_0/bias/v/Read/ReadVariableOpReadVariableOp,Adam/tcn_17/residual_block_1/conv1D_0/bias/v*
_output_shapes	
:*
dtype0
Ύ
.Adam/tcn_17/residual_block_1/conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/tcn_17/residual_block_1/conv1D_1/kernel/v
·
BAdam/tcn_17/residual_block_1/conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/tcn_17/residual_block_1/conv1D_1/kernel/v*$
_output_shapes
:@*
dtype0
±
,Adam/tcn_17/residual_block_1/conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/tcn_17/residual_block_1/conv1D_1/bias/v
ͺ
@Adam/tcn_17/residual_block_1/conv1D_1/bias/v/Read/ReadVariableOpReadVariableOp,Adam/tcn_17/residual_block_1/conv1D_1/bias/v*
_output_shapes	
:*
dtype0
Ύ
.Adam/tcn_17/residual_block_2/conv1D_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/tcn_17/residual_block_2/conv1D_0/kernel/v
·
BAdam/tcn_17/residual_block_2/conv1D_0/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/tcn_17/residual_block_2/conv1D_0/kernel/v*$
_output_shapes
:@*
dtype0
±
,Adam/tcn_17/residual_block_2/conv1D_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/tcn_17/residual_block_2/conv1D_0/bias/v
ͺ
@Adam/tcn_17/residual_block_2/conv1D_0/bias/v/Read/ReadVariableOpReadVariableOp,Adam/tcn_17/residual_block_2/conv1D_0/bias/v*
_output_shapes	
:*
dtype0
Ύ
.Adam/tcn_17/residual_block_2/conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/tcn_17/residual_block_2/conv1D_1/kernel/v
·
BAdam/tcn_17/residual_block_2/conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/tcn_17/residual_block_2/conv1D_1/kernel/v*$
_output_shapes
:@*
dtype0
±
,Adam/tcn_17/residual_block_2/conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/tcn_17/residual_block_2/conv1D_1/bias/v
ͺ
@Adam/tcn_17/residual_block_2/conv1D_1/bias/v/Read/ReadVariableOpReadVariableOp,Adam/tcn_17/residual_block_2/conv1D_1/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ό
value±B­ B₯
ζ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
τ

	dilations
skip_connections
residual_blocks
layers_outputs
residual_block_0
residual_block_1
residual_block_2
slicer_layer
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
¨
"iter

#beta_1

$beta_2
	%decay
&learning_ratemmmm'm(m)m*m+m,m-m.m/m0m1m2m3m4mvvvv'v(v)v*v+v ,v‘-v’.v£/v€0v₯1v¦2v§3v¨4v©

'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
14
15
16
17

'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
14
15
16
17
 
­
5layer_regularization_losses
	variables
6layer_metrics
7metrics

8layers
9non_trainable_variables
trainable_variables
regularization_losses
 
 
 

0
1
2
 
Υ

:layers
;layers_outputs
<shape_match_conv
=final_activation
>conv1D_0
?activation_380
@spatial_dropout1d_190
Aconv1D_1
Bactivation_381
Cspatial_dropout1d_191
Dactivation_382
<matching_conv1D
=activation_383
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
Χ

Ilayers
Jlayers_outputs
Kshape_match_conv
Lfinal_activation
Mconv1D_0
Nactivation_384
Ospatial_dropout1d_192
Pconv1D_1
Qactivation_385
Rspatial_dropout1d_193
Sactivation_386
Kmatching_identity
Lactivation_387
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
Χ

Xlayers
Ylayers_outputs
Zshape_match_conv
[final_activation
\conv1D_0
]activation_388
^spatial_dropout1d_194
_conv1D_1
`activation_389
aspatial_dropout1d_195
bactivation_390
Zmatching_identity
[activation_391
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
R
g	variables
htrainable_variables
iregularization_losses
j	keras_api
f
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
f
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
 
­
klayer_regularization_losses
	variables
llayer_metrics
mmetrics

nlayers
onon_trainable_variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_34/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_34/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
player_regularization_losses
	variables
qlayer_metrics
rmetrics

slayers
tnon_trainable_variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEdense_35/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_35/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
ulayer_regularization_losses
	variables
vlayer_metrics
wmetrics

xlayers
ynon_trainable_variables
trainable_variables
 regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'tcn_17/residual_block_0/conv1D_0/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%tcn_17/residual_block_0/conv1D_0/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'tcn_17/residual_block_0/conv1D_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%tcn_17/residual_block_0/conv1D_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE.tcn_17/residual_block_0/matching_conv1D/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE,tcn_17/residual_block_0/matching_conv1D/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'tcn_17/residual_block_1/conv1D_0/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%tcn_17/residual_block_1/conv1D_0/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'tcn_17/residual_block_1/conv1D_1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%tcn_17/residual_block_1/conv1D_1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'tcn_17/residual_block_2/conv1D_0/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%tcn_17/residual_block_2/conv1D_0/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'tcn_17/residual_block_2/conv1D_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%tcn_17/residual_block_2/conv1D_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
 
 

z0

0
1
2
 
1
>0
?1
@2
A3
B4
C5
D6
 
h

+kernel
,bias
{	variables
|trainable_variables
}regularization_losses
~	keras_api
U
	variables
trainable_variables
regularization_losses
	keras_api
l

'kernel
(bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
l

)kernel
*bias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
*
'0
(1
)2
*3
+4
,5
*
'0
(1
)2
*3
+4
,5
 
²
 layer_regularization_losses
E	variables
 layer_metrics
‘metrics
’layers
£non_trainable_variables
Ftrainable_variables
Gregularization_losses
1
M0
N1
O2
P3
Q4
R5
S6
 
V
€	variables
₯trainable_variables
¦regularization_losses
§	keras_api
V
¨	variables
©trainable_variables
ͺregularization_losses
«	keras_api
l

-kernel
.bias
¬	variables
­trainable_variables
?regularization_losses
―	keras_api
V
°	variables
±trainable_variables
²regularization_losses
³	keras_api
V
΄	variables
΅trainable_variables
Άregularization_losses
·	keras_api
l

/kernel
0bias
Έ	variables
Ήtrainable_variables
Ίregularization_losses
»	keras_api
V
Ό	variables
½trainable_variables
Ύregularization_losses
Ώ	keras_api
V
ΐ	variables
Αtrainable_variables
Βregularization_losses
Γ	keras_api
V
Δ	variables
Εtrainable_variables
Ζregularization_losses
Η	keras_api

-0
.1
/2
03

-0
.1
/2
03
 
²
 Θlayer_regularization_losses
T	variables
Ιlayer_metrics
Κmetrics
Λlayers
Μnon_trainable_variables
Utrainable_variables
Vregularization_losses
1
\0
]1
^2
_3
`4
a5
b6
 
V
Ν	variables
Ξtrainable_variables
Οregularization_losses
Π	keras_api
V
Ρ	variables
?trainable_variables
Σregularization_losses
Τ	keras_api
l

1kernel
2bias
Υ	variables
Φtrainable_variables
Χregularization_losses
Ψ	keras_api
V
Ω	variables
Ϊtrainable_variables
Ϋregularization_losses
ά	keras_api
V
έ	variables
ήtrainable_variables
ίregularization_losses
ΰ	keras_api
l

3kernel
4bias
α	variables
βtrainable_variables
γregularization_losses
δ	keras_api
V
ε	variables
ζtrainable_variables
ηregularization_losses
θ	keras_api
V
ι	variables
κtrainable_variables
λregularization_losses
μ	keras_api
V
ν	variables
ξtrainable_variables
οregularization_losses
π	keras_api

10
21
32
43

10
21
32
43
 
²
 ρlayer_regularization_losses
c	variables
ςlayer_metrics
σmetrics
τlayers
υnon_trainable_variables
dtrainable_variables
eregularization_losses
 
 
 
²
 φlayer_regularization_losses
g	variables
χlayer_metrics
ψmetrics
ωlayers
ϊnon_trainable_variables
htrainable_variables
iregularization_losses
 
 
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
8

ϋtotal

όcount
ύ	variables
ώ	keras_api

+0
,1

+0
,1
 
²
 ?layer_regularization_losses
{	variables
layer_metrics
metrics
layers
non_trainable_variables
|trainable_variables
}regularization_losses
 
 
 
΄
 layer_regularization_losses
	variables
layer_metrics
metrics
layers
non_trainable_variables
trainable_variables
regularization_losses

'0
(1

'0
(1
 
΅
 layer_regularization_losses
	variables
layer_metrics
metrics
layers
non_trainable_variables
trainable_variables
regularization_losses
 
 
 
΅
 layer_regularization_losses
	variables
layer_metrics
metrics
layers
non_trainable_variables
trainable_variables
regularization_losses
 
 
 
΅
 layer_regularization_losses
	variables
layer_metrics
metrics
layers
non_trainable_variables
trainable_variables
regularization_losses

)0
*1

)0
*1
 
΅
 layer_regularization_losses
	variables
layer_metrics
metrics
layers
non_trainable_variables
trainable_variables
regularization_losses
 
 
 
΅
 layer_regularization_losses
	variables
layer_metrics
metrics
 layers
‘non_trainable_variables
trainable_variables
regularization_losses
 
 
 
΅
 ’layer_regularization_losses
	variables
£layer_metrics
€metrics
₯layers
¦non_trainable_variables
trainable_variables
regularization_losses
 
 
 
΅
 §layer_regularization_losses
	variables
¨layer_metrics
©metrics
ͺlayers
«non_trainable_variables
trainable_variables
regularization_losses
 
 
 
?
>0
?1
@2
A3
B4
C5
D6
<7
=8
 
 
 
 
΅
 ¬layer_regularization_losses
€	variables
­layer_metrics
?metrics
―layers
°non_trainable_variables
₯trainable_variables
¦regularization_losses
 
 
 
΅
 ±layer_regularization_losses
¨	variables
²layer_metrics
³metrics
΄layers
΅non_trainable_variables
©trainable_variables
ͺregularization_losses

-0
.1

-0
.1
 
΅
 Άlayer_regularization_losses
¬	variables
·layer_metrics
Έmetrics
Ήlayers
Ίnon_trainable_variables
­trainable_variables
?regularization_losses
 
 
 
΅
 »layer_regularization_losses
°	variables
Όlayer_metrics
½metrics
Ύlayers
Ώnon_trainable_variables
±trainable_variables
²regularization_losses
 
 
 
΅
 ΐlayer_regularization_losses
΄	variables
Αlayer_metrics
Βmetrics
Γlayers
Δnon_trainable_variables
΅trainable_variables
Άregularization_losses

/0
01

/0
01
 
΅
 Εlayer_regularization_losses
Έ	variables
Ζlayer_metrics
Ηmetrics
Θlayers
Ιnon_trainable_variables
Ήtrainable_variables
Ίregularization_losses
 
 
 
΅
 Κlayer_regularization_losses
Ό	variables
Λlayer_metrics
Μmetrics
Νlayers
Ξnon_trainable_variables
½trainable_variables
Ύregularization_losses
 
 
 
΅
 Οlayer_regularization_losses
ΐ	variables
Πlayer_metrics
Ρmetrics
?layers
Σnon_trainable_variables
Αtrainable_variables
Βregularization_losses
 
 
 
΅
 Τlayer_regularization_losses
Δ	variables
Υlayer_metrics
Φmetrics
Χlayers
Ψnon_trainable_variables
Εtrainable_variables
Ζregularization_losses
 
 
 
?
M0
N1
O2
P3
Q4
R5
S6
K7
L8
 
 
 
 
΅
 Ωlayer_regularization_losses
Ν	variables
Ϊlayer_metrics
Ϋmetrics
άlayers
έnon_trainable_variables
Ξtrainable_variables
Οregularization_losses
 
 
 
΅
 ήlayer_regularization_losses
Ρ	variables
ίlayer_metrics
ΰmetrics
αlayers
βnon_trainable_variables
?trainable_variables
Σregularization_losses

10
21

10
21
 
΅
 γlayer_regularization_losses
Υ	variables
δlayer_metrics
εmetrics
ζlayers
ηnon_trainable_variables
Φtrainable_variables
Χregularization_losses
 
 
 
΅
 θlayer_regularization_losses
Ω	variables
ιlayer_metrics
κmetrics
λlayers
μnon_trainable_variables
Ϊtrainable_variables
Ϋregularization_losses
 
 
 
΅
 νlayer_regularization_losses
έ	variables
ξlayer_metrics
οmetrics
πlayers
ρnon_trainable_variables
ήtrainable_variables
ίregularization_losses

30
41

30
41
 
΅
 ςlayer_regularization_losses
α	variables
σlayer_metrics
τmetrics
υlayers
φnon_trainable_variables
βtrainable_variables
γregularization_losses
 
 
 
΅
 χlayer_regularization_losses
ε	variables
ψlayer_metrics
ωmetrics
ϊlayers
ϋnon_trainable_variables
ζtrainable_variables
ηregularization_losses
 
 
 
΅
 όlayer_regularization_losses
ι	variables
ύlayer_metrics
ώmetrics
?layers
non_trainable_variables
κtrainable_variables
λregularization_losses
 
 
 
΅
 layer_regularization_losses
ν	variables
layer_metrics
metrics
layers
non_trainable_variables
ξtrainable_variables
οregularization_losses
 
 
 
?
\0
]1
^2
_3
`4
a5
b6
Z7
[8
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ϋ0
ό1

ύ	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
~|
VARIABLE_VALUEAdam/dense_34/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_34/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_35/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_35/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/tcn_17/residual_block_0/conv1D_0/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/tcn_17/residual_block_0/conv1D_0/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/tcn_17/residual_block_0/conv1D_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/tcn_17/residual_block_0/conv1D_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE5Adam/tcn_17/residual_block_0/matching_conv1D/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE3Adam/tcn_17/residual_block_0/matching_conv1D/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/tcn_17/residual_block_1/conv1D_0/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/tcn_17/residual_block_1/conv1D_0/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/tcn_17/residual_block_1/conv1D_1/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/tcn_17/residual_block_1/conv1D_1/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/tcn_17/residual_block_2/conv1D_0/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/tcn_17/residual_block_2/conv1D_0/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/tcn_17/residual_block_2/conv1D_1/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/tcn_17/residual_block_2/conv1D_1/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_34/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_34/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_35/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_35/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/tcn_17/residual_block_0/conv1D_0/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/tcn_17/residual_block_0/conv1D_0/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/tcn_17/residual_block_0/conv1D_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/tcn_17/residual_block_0/conv1D_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE5Adam/tcn_17/residual_block_0/matching_conv1D/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE3Adam/tcn_17/residual_block_0/matching_conv1D/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/tcn_17/residual_block_1/conv1D_0/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/tcn_17/residual_block_1/conv1D_0/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/tcn_17/residual_block_1/conv1D_1/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/tcn_17/residual_block_1/conv1D_1/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/tcn_17/residual_block_2/conv1D_0/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/tcn_17/residual_block_2/conv1D_0/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/tcn_17/residual_block_2/conv1D_1/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/tcn_17/residual_block_2/conv1D_1/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_tcn_17_inputPlaceholder*,
_output_shapes
:?????????*
dtype0*!
shape:?????????
ή
StatefulPartitionedCallStatefulPartitionedCallserving_default_tcn_17_input'tcn_17/residual_block_0/conv1D_0/kernel%tcn_17/residual_block_0/conv1D_0/bias'tcn_17/residual_block_0/conv1D_1/kernel%tcn_17/residual_block_0/conv1D_1/bias.tcn_17/residual_block_0/matching_conv1D/kernel,tcn_17/residual_block_0/matching_conv1D/bias'tcn_17/residual_block_1/conv1D_0/kernel%tcn_17/residual_block_1/conv1D_0/bias'tcn_17/residual_block_1/conv1D_1/kernel%tcn_17/residual_block_1/conv1D_1/bias'tcn_17/residual_block_2/conv1D_0/kernel%tcn_17/residual_block_2/conv1D_0/bias'tcn_17/residual_block_2/conv1D_1/kernel%tcn_17/residual_block_2/conv1D_1/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1156162
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ι
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOp#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp;tcn_17/residual_block_0/conv1D_0/kernel/Read/ReadVariableOp9tcn_17/residual_block_0/conv1D_0/bias/Read/ReadVariableOp;tcn_17/residual_block_0/conv1D_1/kernel/Read/ReadVariableOp9tcn_17/residual_block_0/conv1D_1/bias/Read/ReadVariableOpBtcn_17/residual_block_0/matching_conv1D/kernel/Read/ReadVariableOp@tcn_17/residual_block_0/matching_conv1D/bias/Read/ReadVariableOp;tcn_17/residual_block_1/conv1D_0/kernel/Read/ReadVariableOp9tcn_17/residual_block_1/conv1D_0/bias/Read/ReadVariableOp;tcn_17/residual_block_1/conv1D_1/kernel/Read/ReadVariableOp9tcn_17/residual_block_1/conv1D_1/bias/Read/ReadVariableOp;tcn_17/residual_block_2/conv1D_0/kernel/Read/ReadVariableOp9tcn_17/residual_block_2/conv1D_0/bias/Read/ReadVariableOp;tcn_17/residual_block_2/conv1D_1/kernel/Read/ReadVariableOp9tcn_17/residual_block_2/conv1D_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_34/kernel/m/Read/ReadVariableOp(Adam/dense_34/bias/m/Read/ReadVariableOp*Adam/dense_35/kernel/m/Read/ReadVariableOp(Adam/dense_35/bias/m/Read/ReadVariableOpBAdam/tcn_17/residual_block_0/conv1D_0/kernel/m/Read/ReadVariableOp@Adam/tcn_17/residual_block_0/conv1D_0/bias/m/Read/ReadVariableOpBAdam/tcn_17/residual_block_0/conv1D_1/kernel/m/Read/ReadVariableOp@Adam/tcn_17/residual_block_0/conv1D_1/bias/m/Read/ReadVariableOpIAdam/tcn_17/residual_block_0/matching_conv1D/kernel/m/Read/ReadVariableOpGAdam/tcn_17/residual_block_0/matching_conv1D/bias/m/Read/ReadVariableOpBAdam/tcn_17/residual_block_1/conv1D_0/kernel/m/Read/ReadVariableOp@Adam/tcn_17/residual_block_1/conv1D_0/bias/m/Read/ReadVariableOpBAdam/tcn_17/residual_block_1/conv1D_1/kernel/m/Read/ReadVariableOp@Adam/tcn_17/residual_block_1/conv1D_1/bias/m/Read/ReadVariableOpBAdam/tcn_17/residual_block_2/conv1D_0/kernel/m/Read/ReadVariableOp@Adam/tcn_17/residual_block_2/conv1D_0/bias/m/Read/ReadVariableOpBAdam/tcn_17/residual_block_2/conv1D_1/kernel/m/Read/ReadVariableOp@Adam/tcn_17/residual_block_2/conv1D_1/bias/m/Read/ReadVariableOp*Adam/dense_34/kernel/v/Read/ReadVariableOp(Adam/dense_34/bias/v/Read/ReadVariableOp*Adam/dense_35/kernel/v/Read/ReadVariableOp(Adam/dense_35/bias/v/Read/ReadVariableOpBAdam/tcn_17/residual_block_0/conv1D_0/kernel/v/Read/ReadVariableOp@Adam/tcn_17/residual_block_0/conv1D_0/bias/v/Read/ReadVariableOpBAdam/tcn_17/residual_block_0/conv1D_1/kernel/v/Read/ReadVariableOp@Adam/tcn_17/residual_block_0/conv1D_1/bias/v/Read/ReadVariableOpIAdam/tcn_17/residual_block_0/matching_conv1D/kernel/v/Read/ReadVariableOpGAdam/tcn_17/residual_block_0/matching_conv1D/bias/v/Read/ReadVariableOpBAdam/tcn_17/residual_block_1/conv1D_0/kernel/v/Read/ReadVariableOp@Adam/tcn_17/residual_block_1/conv1D_0/bias/v/Read/ReadVariableOpBAdam/tcn_17/residual_block_1/conv1D_1/kernel/v/Read/ReadVariableOp@Adam/tcn_17/residual_block_1/conv1D_1/bias/v/Read/ReadVariableOpBAdam/tcn_17/residual_block_2/conv1D_0/kernel/v/Read/ReadVariableOp@Adam/tcn_17/residual_block_2/conv1D_0/bias/v/Read/ReadVariableOpBAdam/tcn_17/residual_block_2/conv1D_1/kernel/v/Read/ReadVariableOp@Adam/tcn_17/residual_block_2/conv1D_1/bias/v/Read/ReadVariableOpConst*J
TinC
A2?	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_1157663
 
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_34/kerneldense_34/biasdense_35/kerneldense_35/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate'tcn_17/residual_block_0/conv1D_0/kernel%tcn_17/residual_block_0/conv1D_0/bias'tcn_17/residual_block_0/conv1D_1/kernel%tcn_17/residual_block_0/conv1D_1/bias.tcn_17/residual_block_0/matching_conv1D/kernel,tcn_17/residual_block_0/matching_conv1D/bias'tcn_17/residual_block_1/conv1D_0/kernel%tcn_17/residual_block_1/conv1D_0/bias'tcn_17/residual_block_1/conv1D_1/kernel%tcn_17/residual_block_1/conv1D_1/bias'tcn_17/residual_block_2/conv1D_0/kernel%tcn_17/residual_block_2/conv1D_0/bias'tcn_17/residual_block_2/conv1D_1/kernel%tcn_17/residual_block_2/conv1D_1/biastotalcountAdam/dense_34/kernel/mAdam/dense_34/bias/mAdam/dense_35/kernel/mAdam/dense_35/bias/m.Adam/tcn_17/residual_block_0/conv1D_0/kernel/m,Adam/tcn_17/residual_block_0/conv1D_0/bias/m.Adam/tcn_17/residual_block_0/conv1D_1/kernel/m,Adam/tcn_17/residual_block_0/conv1D_1/bias/m5Adam/tcn_17/residual_block_0/matching_conv1D/kernel/m3Adam/tcn_17/residual_block_0/matching_conv1D/bias/m.Adam/tcn_17/residual_block_1/conv1D_0/kernel/m,Adam/tcn_17/residual_block_1/conv1D_0/bias/m.Adam/tcn_17/residual_block_1/conv1D_1/kernel/m,Adam/tcn_17/residual_block_1/conv1D_1/bias/m.Adam/tcn_17/residual_block_2/conv1D_0/kernel/m,Adam/tcn_17/residual_block_2/conv1D_0/bias/m.Adam/tcn_17/residual_block_2/conv1D_1/kernel/m,Adam/tcn_17/residual_block_2/conv1D_1/bias/mAdam/dense_34/kernel/vAdam/dense_34/bias/vAdam/dense_35/kernel/vAdam/dense_35/bias/v.Adam/tcn_17/residual_block_0/conv1D_0/kernel/v,Adam/tcn_17/residual_block_0/conv1D_0/bias/v.Adam/tcn_17/residual_block_0/conv1D_1/kernel/v,Adam/tcn_17/residual_block_0/conv1D_1/bias/v5Adam/tcn_17/residual_block_0/matching_conv1D/kernel/v3Adam/tcn_17/residual_block_0/matching_conv1D/bias/v.Adam/tcn_17/residual_block_1/conv1D_0/kernel/v,Adam/tcn_17/residual_block_1/conv1D_0/bias/v.Adam/tcn_17/residual_block_1/conv1D_1/kernel/v,Adam/tcn_17/residual_block_1/conv1D_1/bias/v.Adam/tcn_17/residual_block_2/conv1D_0/kernel/v,Adam/tcn_17/residual_block_2/conv1D_0/bias/v.Adam/tcn_17/residual_block_2/conv1D_1/kernel/v,Adam/tcn_17/residual_block_2/conv1D_1/bias/v*I
TinB
@2>*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1157856ΪΫ

S
7__inference_spatial_dropout1d_194_layer_call_fn_1157420

inputs
identityι
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_spatial_dropout1d_194_layer_call_and_return_conditional_losses_11552452
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
ζ

χ
%__inference_signature_wrapper_1156162
tcn_17_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity’StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCalltcn_17_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_11549182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
,
_output_shapes
:?????????
&
_user_specified_nametcn_17_input

ϋ
/__inference_sequential_17_layer_call_fn_1156659

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity’StatefulPartitionedCallΥ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_11559882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
­
p
R__inference_spatial_dropout1d_191_layer_call_and_return_conditional_losses_1157299

inputs

identity_1p
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????2

Identity

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
­
p
R__inference_spatial_dropout1d_190_layer_call_and_return_conditional_losses_1154981

inputs

identity_1p
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????2

Identity

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
π	
΄
(__inference_tcn_17_layer_call_fn_1157163

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_tcn_17_layer_call_and_return_conditional_losses_11555842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:?????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
?
q
R__inference_spatial_dropout1d_190_layer_call_and_return_conditional_losses_1157257

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1Ν
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeΠ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2
dropout/GreaterEqual/yΛ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul_1{
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs

S
7__inference_spatial_dropout1d_195_layer_call_fn_1157457

inputs
identityι
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_spatial_dropout1d_195_layer_call_and_return_conditional_losses_11553112
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
Ά€
Υ

J__inference_sequential_17_layer_call_and_return_conditional_losses_1156441

inputsP
Ltcn_17_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resourceD
@tcn_17_residual_block_0_conv1d_0_biasadd_readvariableop_resourceP
Ltcn_17_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resourceD
@tcn_17_residual_block_0_conv1d_1_biasadd_readvariableop_resourceW
Stcn_17_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resourceK
Gtcn_17_residual_block_0_matching_conv1d_biasadd_readvariableop_resourceP
Ltcn_17_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resourceD
@tcn_17_residual_block_1_conv1d_0_biasadd_readvariableop_resourceP
Ltcn_17_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resourceD
@tcn_17_residual_block_1_conv1d_1_biasadd_readvariableop_resourceP
Ltcn_17_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resourceD
@tcn_17_residual_block_2_conv1d_0_biasadd_readvariableop_resourceP
Ltcn_17_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resourceD
@tcn_17_residual_block_2_conv1d_1_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identityΗ
-tcn_17/residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ?               2/
-tcn_17/residual_block_0/conv1D_0/Pad/paddingsΚ
$tcn_17/residual_block_0/conv1D_0/PadPadinputs6tcn_17/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:?????????Ώ2&
$tcn_17/residual_block_0/conv1D_0/Pad»
6tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????28
6tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims/dim‘
2tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims
ExpandDims-tcn_17/residual_block_0/conv1D_0/Pad:output:0?tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Ώ24
2tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims
Ctcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpLtcn_17_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02E
Ctcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpΆ
8tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/dimΌ
4tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1
ExpandDimsKtcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0Atcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@26
4tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1½
'tcn_17/residual_block_0/conv1D_0/conv1dConv2D;tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims:output:0=tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2)
'tcn_17/residual_block_0/conv1D_0/conv1dχ
/tcn_17/residual_block_0/conv1D_0/conv1d/SqueezeSqueeze0tcn_17/residual_block_0/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????21
/tcn_17/residual_block_0/conv1D_0/conv1d/Squeezeπ
7tcn_17/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp@tcn_17_residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7tcn_17/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp
(tcn_17/residual_block_0/conv1D_0/BiasAddBiasAdd8tcn_17/residual_block_0/conv1D_0/conv1d/Squeeze:output:0?tcn_17/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2*
(tcn_17/residual_block_0/conv1D_0/BiasAddΝ
+tcn_17/residual_block_0/activation_380/ReluRelu1tcn_17/residual_block_0/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_0/activation_380/ReluΣ
3tcn_17/residual_block_0/spatial_dropout1d_190/ShapeShape9tcn_17/residual_block_0/activation_380/Relu:activations:0*
T0*
_output_shapes
:25
3tcn_17/residual_block_0/spatial_dropout1d_190/ShapeΠ
Atcn_17/residual_block_0/spatial_dropout1d_190/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Atcn_17/residual_block_0/spatial_dropout1d_190/strided_slice/stackΤ
Ctcn_17/residual_block_0/spatial_dropout1d_190/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_0/spatial_dropout1d_190/strided_slice/stack_1Τ
Ctcn_17/residual_block_0/spatial_dropout1d_190/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_0/spatial_dropout1d_190/strided_slice/stack_2φ
;tcn_17/residual_block_0/spatial_dropout1d_190/strided_sliceStridedSlice<tcn_17/residual_block_0/spatial_dropout1d_190/Shape:output:0Jtcn_17/residual_block_0/spatial_dropout1d_190/strided_slice/stack:output:0Ltcn_17/residual_block_0/spatial_dropout1d_190/strided_slice/stack_1:output:0Ltcn_17/residual_block_0/spatial_dropout1d_190/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;tcn_17/residual_block_0/spatial_dropout1d_190/strided_sliceΤ
Ctcn_17/residual_block_0/spatial_dropout1d_190/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_0/spatial_dropout1d_190/strided_slice_1/stackΨ
Etcn_17/residual_block_0/spatial_dropout1d_190/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Etcn_17/residual_block_0/spatial_dropout1d_190/strided_slice_1/stack_1Ψ
Etcn_17/residual_block_0/spatial_dropout1d_190/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Etcn_17/residual_block_0/spatial_dropout1d_190/strided_slice_1/stack_2
=tcn_17/residual_block_0/spatial_dropout1d_190/strided_slice_1StridedSlice<tcn_17/residual_block_0/spatial_dropout1d_190/Shape:output:0Ltcn_17/residual_block_0/spatial_dropout1d_190/strided_slice_1/stack:output:0Ntcn_17/residual_block_0/spatial_dropout1d_190/strided_slice_1/stack_1:output:0Ntcn_17/residual_block_0/spatial_dropout1d_190/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=tcn_17/residual_block_0/spatial_dropout1d_190/strided_slice_1Ώ
;tcn_17/residual_block_0/spatial_dropout1d_190/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2=
;tcn_17/residual_block_0/spatial_dropout1d_190/dropout/ConstΆ
9tcn_17/residual_block_0/spatial_dropout1d_190/dropout/MulMul9tcn_17/residual_block_0/activation_380/Relu:activations:0Dtcn_17/residual_block_0/spatial_dropout1d_190/dropout/Const:output:0*
T0*-
_output_shapes
:?????????2;
9tcn_17/residual_block_0/spatial_dropout1d_190/dropout/Mulή
Ltcn_17/residual_block_0/spatial_dropout1d_190/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2N
Ltcn_17/residual_block_0/spatial_dropout1d_190/dropout/random_uniform/shape/1³
Jtcn_17/residual_block_0/spatial_dropout1d_190/dropout/random_uniform/shapePackDtcn_17/residual_block_0/spatial_dropout1d_190/strided_slice:output:0Utcn_17/residual_block_0/spatial_dropout1d_190/dropout/random_uniform/shape/1:output:0Ftcn_17/residual_block_0/spatial_dropout1d_190/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2L
Jtcn_17/residual_block_0/spatial_dropout1d_190/dropout/random_uniform/shapeΪ
Rtcn_17/residual_block_0/spatial_dropout1d_190/dropout/random_uniform/RandomUniformRandomUniformStcn_17/residual_block_0/spatial_dropout1d_190/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02T
Rtcn_17/residual_block_0/spatial_dropout1d_190/dropout/random_uniform/RandomUniformΡ
Dtcn_17/residual_block_0/spatial_dropout1d_190/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2F
Dtcn_17/residual_block_0/spatial_dropout1d_190/dropout/GreaterEqual/y
Btcn_17/residual_block_0/spatial_dropout1d_190/dropout/GreaterEqualGreaterEqual[tcn_17/residual_block_0/spatial_dropout1d_190/dropout/random_uniform/RandomUniform:output:0Mtcn_17/residual_block_0/spatial_dropout1d_190/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2D
Btcn_17/residual_block_0/spatial_dropout1d_190/dropout/GreaterEqual
:tcn_17/residual_block_0/spatial_dropout1d_190/dropout/CastCastFtcn_17/residual_block_0/spatial_dropout1d_190/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2<
:tcn_17/residual_block_0/spatial_dropout1d_190/dropout/CastΈ
;tcn_17/residual_block_0/spatial_dropout1d_190/dropout/Mul_1Mul=tcn_17/residual_block_0/spatial_dropout1d_190/dropout/Mul:z:0>tcn_17/residual_block_0/spatial_dropout1d_190/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????2=
;tcn_17/residual_block_0/spatial_dropout1d_190/dropout/Mul_1Η
-tcn_17/residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ?               2/
-tcn_17/residual_block_0/conv1D_1/Pad/paddings
$tcn_17/residual_block_0/conv1D_1/PadPad?tcn_17/residual_block_0/spatial_dropout1d_190/dropout/Mul_1:z:06tcn_17/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ2&
$tcn_17/residual_block_0/conv1D_1/Pad»
6tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????28
6tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims/dim’
2tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims
ExpandDims-tcn_17/residual_block_0/conv1D_1/Pad:output:0?tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ24
2tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims
Ctcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpLtcn_17_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02E
Ctcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpΆ
8tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/dim½
4tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1
ExpandDimsKtcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Atcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@26
4tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1½
'tcn_17/residual_block_0/conv1D_1/conv1dConv2D;tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims:output:0=tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2)
'tcn_17/residual_block_0/conv1D_1/conv1dχ
/tcn_17/residual_block_0/conv1D_1/conv1d/SqueezeSqueeze0tcn_17/residual_block_0/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????21
/tcn_17/residual_block_0/conv1D_1/conv1d/Squeezeπ
7tcn_17/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp@tcn_17_residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7tcn_17/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp
(tcn_17/residual_block_0/conv1D_1/BiasAddBiasAdd8tcn_17/residual_block_0/conv1D_1/conv1d/Squeeze:output:0?tcn_17/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2*
(tcn_17/residual_block_0/conv1D_1/BiasAddΝ
+tcn_17/residual_block_0/activation_381/ReluRelu1tcn_17/residual_block_0/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_0/activation_381/ReluΣ
3tcn_17/residual_block_0/spatial_dropout1d_191/ShapeShape9tcn_17/residual_block_0/activation_381/Relu:activations:0*
T0*
_output_shapes
:25
3tcn_17/residual_block_0/spatial_dropout1d_191/ShapeΠ
Atcn_17/residual_block_0/spatial_dropout1d_191/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Atcn_17/residual_block_0/spatial_dropout1d_191/strided_slice/stackΤ
Ctcn_17/residual_block_0/spatial_dropout1d_191/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_0/spatial_dropout1d_191/strided_slice/stack_1Τ
Ctcn_17/residual_block_0/spatial_dropout1d_191/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_0/spatial_dropout1d_191/strided_slice/stack_2φ
;tcn_17/residual_block_0/spatial_dropout1d_191/strided_sliceStridedSlice<tcn_17/residual_block_0/spatial_dropout1d_191/Shape:output:0Jtcn_17/residual_block_0/spatial_dropout1d_191/strided_slice/stack:output:0Ltcn_17/residual_block_0/spatial_dropout1d_191/strided_slice/stack_1:output:0Ltcn_17/residual_block_0/spatial_dropout1d_191/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;tcn_17/residual_block_0/spatial_dropout1d_191/strided_sliceΤ
Ctcn_17/residual_block_0/spatial_dropout1d_191/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_0/spatial_dropout1d_191/strided_slice_1/stackΨ
Etcn_17/residual_block_0/spatial_dropout1d_191/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Etcn_17/residual_block_0/spatial_dropout1d_191/strided_slice_1/stack_1Ψ
Etcn_17/residual_block_0/spatial_dropout1d_191/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Etcn_17/residual_block_0/spatial_dropout1d_191/strided_slice_1/stack_2
=tcn_17/residual_block_0/spatial_dropout1d_191/strided_slice_1StridedSlice<tcn_17/residual_block_0/spatial_dropout1d_191/Shape:output:0Ltcn_17/residual_block_0/spatial_dropout1d_191/strided_slice_1/stack:output:0Ntcn_17/residual_block_0/spatial_dropout1d_191/strided_slice_1/stack_1:output:0Ntcn_17/residual_block_0/spatial_dropout1d_191/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=tcn_17/residual_block_0/spatial_dropout1d_191/strided_slice_1Ώ
;tcn_17/residual_block_0/spatial_dropout1d_191/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2=
;tcn_17/residual_block_0/spatial_dropout1d_191/dropout/ConstΆ
9tcn_17/residual_block_0/spatial_dropout1d_191/dropout/MulMul9tcn_17/residual_block_0/activation_381/Relu:activations:0Dtcn_17/residual_block_0/spatial_dropout1d_191/dropout/Const:output:0*
T0*-
_output_shapes
:?????????2;
9tcn_17/residual_block_0/spatial_dropout1d_191/dropout/Mulή
Ltcn_17/residual_block_0/spatial_dropout1d_191/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2N
Ltcn_17/residual_block_0/spatial_dropout1d_191/dropout/random_uniform/shape/1³
Jtcn_17/residual_block_0/spatial_dropout1d_191/dropout/random_uniform/shapePackDtcn_17/residual_block_0/spatial_dropout1d_191/strided_slice:output:0Utcn_17/residual_block_0/spatial_dropout1d_191/dropout/random_uniform/shape/1:output:0Ftcn_17/residual_block_0/spatial_dropout1d_191/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2L
Jtcn_17/residual_block_0/spatial_dropout1d_191/dropout/random_uniform/shapeΪ
Rtcn_17/residual_block_0/spatial_dropout1d_191/dropout/random_uniform/RandomUniformRandomUniformStcn_17/residual_block_0/spatial_dropout1d_191/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02T
Rtcn_17/residual_block_0/spatial_dropout1d_191/dropout/random_uniform/RandomUniformΡ
Dtcn_17/residual_block_0/spatial_dropout1d_191/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2F
Dtcn_17/residual_block_0/spatial_dropout1d_191/dropout/GreaterEqual/y
Btcn_17/residual_block_0/spatial_dropout1d_191/dropout/GreaterEqualGreaterEqual[tcn_17/residual_block_0/spatial_dropout1d_191/dropout/random_uniform/RandomUniform:output:0Mtcn_17/residual_block_0/spatial_dropout1d_191/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2D
Btcn_17/residual_block_0/spatial_dropout1d_191/dropout/GreaterEqual
:tcn_17/residual_block_0/spatial_dropout1d_191/dropout/CastCastFtcn_17/residual_block_0/spatial_dropout1d_191/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2<
:tcn_17/residual_block_0/spatial_dropout1d_191/dropout/CastΈ
;tcn_17/residual_block_0/spatial_dropout1d_191/dropout/Mul_1Mul=tcn_17/residual_block_0/spatial_dropout1d_191/dropout/Mul:z:0>tcn_17/residual_block_0/spatial_dropout1d_191/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????2=
;tcn_17/residual_block_0/spatial_dropout1d_191/dropout/Mul_1Ϋ
+tcn_17/residual_block_0/activation_382/ReluRelu?tcn_17/residual_block_0/spatial_dropout1d_191/dropout/Mul_1:z:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_0/activation_382/ReluΙ
=tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2?
=tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims/dim
9tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims
ExpandDimsinputsFtcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2;
9tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims±
Jtcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpStcn_17_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:*
dtype02L
Jtcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOpΔ
?tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2A
?tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/dimΨ
;tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1
ExpandDimsRtcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOp:value:0Htcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:2=
;tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1Ψ
.tcn_17/residual_block_0/matching_conv1D/conv1dConv2DBtcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims:output:0Dtcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingSAME*
strides
20
.tcn_17/residual_block_0/matching_conv1D/conv1d
6tcn_17/residual_block_0/matching_conv1D/conv1d/SqueezeSqueeze7tcn_17/residual_block_0/matching_conv1D/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????28
6tcn_17/residual_block_0/matching_conv1D/conv1d/Squeeze
>tcn_17/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOpGtcn_17_residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02@
>tcn_17/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp?
/tcn_17/residual_block_0/matching_conv1D/BiasAddBiasAdd?tcn_17/residual_block_0/matching_conv1D/conv1d/Squeeze:output:0Ftcn_17/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????21
/tcn_17/residual_block_0/matching_conv1D/BiasAddψ
tcn_17/residual_block_0/add/addAddV28tcn_17/residual_block_0/matching_conv1D/BiasAdd:output:09tcn_17/residual_block_0/activation_382/Relu:activations:0*
T0*-
_output_shapes
:?????????2!
tcn_17/residual_block_0/add/addΏ
+tcn_17/residual_block_0/activation_383/ReluRelu#tcn_17/residual_block_0/add/add:z:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_0/activation_383/ReluΗ
-tcn_17/residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ~               2/
-tcn_17/residual_block_1/conv1D_0/Pad/paddingsώ
$tcn_17/residual_block_1/conv1D_0/PadPad9tcn_17/residual_block_0/activation_383/Relu:activations:06tcn_17/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ώ2&
$tcn_17/residual_block_1/conv1D_0/PadΈ
5tcn_17/residual_block_1/conv1D_0/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:27
5tcn_17/residual_block_1/conv1D_0/conv1d/dilation_rateχ
Ttcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ώ2V
Ttcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/input_shape
Vtcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2X
Vtcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddings?
Qtcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2S
Qtcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsω
Ntcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2P
Ntcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/crops?
Btcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Btcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shapeΫ
?tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2A
?tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddings
6tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchNDSpaceToBatchND-tcn_17/residual_block_1/conv1D_0/Pad:output:0Ktcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shape:output:0Htcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ28
6tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND»
6tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????28
6tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims/dim΄
2tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims
ExpandDims?tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND:output:0?tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ24
2tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims
Ctcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpLtcn_17_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02E
Ctcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpΆ
8tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/dim½
4tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1
ExpandDimsKtcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0Atcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@26
4tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1½
'tcn_17/residual_block_1/conv1D_0/conv1dConv2D;tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims:output:0=tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2)
'tcn_17/residual_block_1/conv1D_0/conv1dχ
/tcn_17/residual_block_1/conv1D_0/conv1d/SqueezeSqueeze0tcn_17/residual_block_1/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????21
/tcn_17/residual_block_1/conv1D_0/conv1d/Squeeze?
Btcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Btcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shapeΥ
<tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2>
<tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/crops
6tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceNDBatchToSpaceND8tcn_17/residual_block_1/conv1D_0/conv1d/Squeeze:output:0Ktcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shape:output:0Etcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????28
6tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceNDπ
7tcn_17/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp@tcn_17_residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7tcn_17/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp
(tcn_17/residual_block_1/conv1D_0/BiasAddBiasAdd?tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND:output:0?tcn_17/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2*
(tcn_17/residual_block_1/conv1D_0/BiasAddΝ
+tcn_17/residual_block_1/activation_384/ReluRelu1tcn_17/residual_block_1/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_1/activation_384/ReluΣ
3tcn_17/residual_block_1/spatial_dropout1d_192/ShapeShape9tcn_17/residual_block_1/activation_384/Relu:activations:0*
T0*
_output_shapes
:25
3tcn_17/residual_block_1/spatial_dropout1d_192/ShapeΠ
Atcn_17/residual_block_1/spatial_dropout1d_192/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Atcn_17/residual_block_1/spatial_dropout1d_192/strided_slice/stackΤ
Ctcn_17/residual_block_1/spatial_dropout1d_192/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_1/spatial_dropout1d_192/strided_slice/stack_1Τ
Ctcn_17/residual_block_1/spatial_dropout1d_192/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_1/spatial_dropout1d_192/strided_slice/stack_2φ
;tcn_17/residual_block_1/spatial_dropout1d_192/strided_sliceStridedSlice<tcn_17/residual_block_1/spatial_dropout1d_192/Shape:output:0Jtcn_17/residual_block_1/spatial_dropout1d_192/strided_slice/stack:output:0Ltcn_17/residual_block_1/spatial_dropout1d_192/strided_slice/stack_1:output:0Ltcn_17/residual_block_1/spatial_dropout1d_192/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;tcn_17/residual_block_1/spatial_dropout1d_192/strided_sliceΤ
Ctcn_17/residual_block_1/spatial_dropout1d_192/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_1/spatial_dropout1d_192/strided_slice_1/stackΨ
Etcn_17/residual_block_1/spatial_dropout1d_192/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Etcn_17/residual_block_1/spatial_dropout1d_192/strided_slice_1/stack_1Ψ
Etcn_17/residual_block_1/spatial_dropout1d_192/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Etcn_17/residual_block_1/spatial_dropout1d_192/strided_slice_1/stack_2
=tcn_17/residual_block_1/spatial_dropout1d_192/strided_slice_1StridedSlice<tcn_17/residual_block_1/spatial_dropout1d_192/Shape:output:0Ltcn_17/residual_block_1/spatial_dropout1d_192/strided_slice_1/stack:output:0Ntcn_17/residual_block_1/spatial_dropout1d_192/strided_slice_1/stack_1:output:0Ntcn_17/residual_block_1/spatial_dropout1d_192/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=tcn_17/residual_block_1/spatial_dropout1d_192/strided_slice_1Ώ
;tcn_17/residual_block_1/spatial_dropout1d_192/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2=
;tcn_17/residual_block_1/spatial_dropout1d_192/dropout/ConstΆ
9tcn_17/residual_block_1/spatial_dropout1d_192/dropout/MulMul9tcn_17/residual_block_1/activation_384/Relu:activations:0Dtcn_17/residual_block_1/spatial_dropout1d_192/dropout/Const:output:0*
T0*-
_output_shapes
:?????????2;
9tcn_17/residual_block_1/spatial_dropout1d_192/dropout/Mulή
Ltcn_17/residual_block_1/spatial_dropout1d_192/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2N
Ltcn_17/residual_block_1/spatial_dropout1d_192/dropout/random_uniform/shape/1³
Jtcn_17/residual_block_1/spatial_dropout1d_192/dropout/random_uniform/shapePackDtcn_17/residual_block_1/spatial_dropout1d_192/strided_slice:output:0Utcn_17/residual_block_1/spatial_dropout1d_192/dropout/random_uniform/shape/1:output:0Ftcn_17/residual_block_1/spatial_dropout1d_192/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2L
Jtcn_17/residual_block_1/spatial_dropout1d_192/dropout/random_uniform/shapeΪ
Rtcn_17/residual_block_1/spatial_dropout1d_192/dropout/random_uniform/RandomUniformRandomUniformStcn_17/residual_block_1/spatial_dropout1d_192/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02T
Rtcn_17/residual_block_1/spatial_dropout1d_192/dropout/random_uniform/RandomUniformΡ
Dtcn_17/residual_block_1/spatial_dropout1d_192/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2F
Dtcn_17/residual_block_1/spatial_dropout1d_192/dropout/GreaterEqual/y
Btcn_17/residual_block_1/spatial_dropout1d_192/dropout/GreaterEqualGreaterEqual[tcn_17/residual_block_1/spatial_dropout1d_192/dropout/random_uniform/RandomUniform:output:0Mtcn_17/residual_block_1/spatial_dropout1d_192/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2D
Btcn_17/residual_block_1/spatial_dropout1d_192/dropout/GreaterEqual
:tcn_17/residual_block_1/spatial_dropout1d_192/dropout/CastCastFtcn_17/residual_block_1/spatial_dropout1d_192/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2<
:tcn_17/residual_block_1/spatial_dropout1d_192/dropout/CastΈ
;tcn_17/residual_block_1/spatial_dropout1d_192/dropout/Mul_1Mul=tcn_17/residual_block_1/spatial_dropout1d_192/dropout/Mul:z:0>tcn_17/residual_block_1/spatial_dropout1d_192/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????2=
;tcn_17/residual_block_1/spatial_dropout1d_192/dropout/Mul_1Η
-tcn_17/residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ~               2/
-tcn_17/residual_block_1/conv1D_1/Pad/paddings
$tcn_17/residual_block_1/conv1D_1/PadPad?tcn_17/residual_block_1/spatial_dropout1d_192/dropout/Mul_1:z:06tcn_17/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ώ2&
$tcn_17/residual_block_1/conv1D_1/PadΈ
5tcn_17/residual_block_1/conv1D_1/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:27
5tcn_17/residual_block_1/conv1D_1/conv1d/dilation_rateχ
Ttcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ώ2V
Ttcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/input_shape
Vtcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2X
Vtcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddings?
Qtcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2S
Qtcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsω
Ntcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2P
Ntcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/crops?
Btcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Btcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shapeΫ
?tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2A
?tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddings
6tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchNDSpaceToBatchND-tcn_17/residual_block_1/conv1D_1/Pad:output:0Ktcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shape:output:0Htcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ28
6tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND»
6tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????28
6tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims/dim΄
2tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims
ExpandDims?tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND:output:0?tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ24
2tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims
Ctcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpLtcn_17_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02E
Ctcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpΆ
8tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/dim½
4tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1
ExpandDimsKtcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Atcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@26
4tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1½
'tcn_17/residual_block_1/conv1D_1/conv1dConv2D;tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims:output:0=tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2)
'tcn_17/residual_block_1/conv1D_1/conv1dχ
/tcn_17/residual_block_1/conv1D_1/conv1d/SqueezeSqueeze0tcn_17/residual_block_1/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????21
/tcn_17/residual_block_1/conv1D_1/conv1d/Squeeze?
Btcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Btcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shapeΥ
<tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2>
<tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/crops
6tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceNDBatchToSpaceND8tcn_17/residual_block_1/conv1D_1/conv1d/Squeeze:output:0Ktcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shape:output:0Etcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????28
6tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceNDπ
7tcn_17/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp@tcn_17_residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7tcn_17/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp
(tcn_17/residual_block_1/conv1D_1/BiasAddBiasAdd?tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND:output:0?tcn_17/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2*
(tcn_17/residual_block_1/conv1D_1/BiasAddΝ
+tcn_17/residual_block_1/activation_385/ReluRelu1tcn_17/residual_block_1/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_1/activation_385/ReluΣ
3tcn_17/residual_block_1/spatial_dropout1d_193/ShapeShape9tcn_17/residual_block_1/activation_385/Relu:activations:0*
T0*
_output_shapes
:25
3tcn_17/residual_block_1/spatial_dropout1d_193/ShapeΠ
Atcn_17/residual_block_1/spatial_dropout1d_193/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Atcn_17/residual_block_1/spatial_dropout1d_193/strided_slice/stackΤ
Ctcn_17/residual_block_1/spatial_dropout1d_193/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_1/spatial_dropout1d_193/strided_slice/stack_1Τ
Ctcn_17/residual_block_1/spatial_dropout1d_193/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_1/spatial_dropout1d_193/strided_slice/stack_2φ
;tcn_17/residual_block_1/spatial_dropout1d_193/strided_sliceStridedSlice<tcn_17/residual_block_1/spatial_dropout1d_193/Shape:output:0Jtcn_17/residual_block_1/spatial_dropout1d_193/strided_slice/stack:output:0Ltcn_17/residual_block_1/spatial_dropout1d_193/strided_slice/stack_1:output:0Ltcn_17/residual_block_1/spatial_dropout1d_193/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;tcn_17/residual_block_1/spatial_dropout1d_193/strided_sliceΤ
Ctcn_17/residual_block_1/spatial_dropout1d_193/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_1/spatial_dropout1d_193/strided_slice_1/stackΨ
Etcn_17/residual_block_1/spatial_dropout1d_193/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Etcn_17/residual_block_1/spatial_dropout1d_193/strided_slice_1/stack_1Ψ
Etcn_17/residual_block_1/spatial_dropout1d_193/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Etcn_17/residual_block_1/spatial_dropout1d_193/strided_slice_1/stack_2
=tcn_17/residual_block_1/spatial_dropout1d_193/strided_slice_1StridedSlice<tcn_17/residual_block_1/spatial_dropout1d_193/Shape:output:0Ltcn_17/residual_block_1/spatial_dropout1d_193/strided_slice_1/stack:output:0Ntcn_17/residual_block_1/spatial_dropout1d_193/strided_slice_1/stack_1:output:0Ntcn_17/residual_block_1/spatial_dropout1d_193/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=tcn_17/residual_block_1/spatial_dropout1d_193/strided_slice_1Ώ
;tcn_17/residual_block_1/spatial_dropout1d_193/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2=
;tcn_17/residual_block_1/spatial_dropout1d_193/dropout/ConstΆ
9tcn_17/residual_block_1/spatial_dropout1d_193/dropout/MulMul9tcn_17/residual_block_1/activation_385/Relu:activations:0Dtcn_17/residual_block_1/spatial_dropout1d_193/dropout/Const:output:0*
T0*-
_output_shapes
:?????????2;
9tcn_17/residual_block_1/spatial_dropout1d_193/dropout/Mulή
Ltcn_17/residual_block_1/spatial_dropout1d_193/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2N
Ltcn_17/residual_block_1/spatial_dropout1d_193/dropout/random_uniform/shape/1³
Jtcn_17/residual_block_1/spatial_dropout1d_193/dropout/random_uniform/shapePackDtcn_17/residual_block_1/spatial_dropout1d_193/strided_slice:output:0Utcn_17/residual_block_1/spatial_dropout1d_193/dropout/random_uniform/shape/1:output:0Ftcn_17/residual_block_1/spatial_dropout1d_193/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2L
Jtcn_17/residual_block_1/spatial_dropout1d_193/dropout/random_uniform/shapeΪ
Rtcn_17/residual_block_1/spatial_dropout1d_193/dropout/random_uniform/RandomUniformRandomUniformStcn_17/residual_block_1/spatial_dropout1d_193/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02T
Rtcn_17/residual_block_1/spatial_dropout1d_193/dropout/random_uniform/RandomUniformΡ
Dtcn_17/residual_block_1/spatial_dropout1d_193/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2F
Dtcn_17/residual_block_1/spatial_dropout1d_193/dropout/GreaterEqual/y
Btcn_17/residual_block_1/spatial_dropout1d_193/dropout/GreaterEqualGreaterEqual[tcn_17/residual_block_1/spatial_dropout1d_193/dropout/random_uniform/RandomUniform:output:0Mtcn_17/residual_block_1/spatial_dropout1d_193/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2D
Btcn_17/residual_block_1/spatial_dropout1d_193/dropout/GreaterEqual
:tcn_17/residual_block_1/spatial_dropout1d_193/dropout/CastCastFtcn_17/residual_block_1/spatial_dropout1d_193/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2<
:tcn_17/residual_block_1/spatial_dropout1d_193/dropout/CastΈ
;tcn_17/residual_block_1/spatial_dropout1d_193/dropout/Mul_1Mul=tcn_17/residual_block_1/spatial_dropout1d_193/dropout/Mul:z:0>tcn_17/residual_block_1/spatial_dropout1d_193/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????2=
;tcn_17/residual_block_1/spatial_dropout1d_193/dropout/Mul_1Ϋ
+tcn_17/residual_block_1/activation_386/ReluRelu?tcn_17/residual_block_1/spatial_dropout1d_193/dropout/Mul_1:z:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_1/activation_386/Reluύ
!tcn_17/residual_block_1/add_1/addAddV29tcn_17/residual_block_0/activation_383/Relu:activations:09tcn_17/residual_block_1/activation_386/Relu:activations:0*
T0*-
_output_shapes
:?????????2#
!tcn_17/residual_block_1/add_1/addΑ
+tcn_17/residual_block_1/activation_387/ReluRelu%tcn_17/residual_block_1/add_1/add:z:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_1/activation_387/ReluΗ
-tcn_17/residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ό               2/
-tcn_17/residual_block_2/conv1D_0/Pad/paddingsώ
$tcn_17/residual_block_2/conv1D_0/PadPad9tcn_17/residual_block_1/activation_387/Relu:activations:06tcn_17/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ό2&
$tcn_17/residual_block_2/conv1D_0/PadΈ
5tcn_17/residual_block_2/conv1D_0/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:27
5tcn_17/residual_block_2/conv1D_0/conv1d/dilation_rateχ
Ttcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ό2V
Ttcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/input_shape
Vtcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2X
Vtcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddings?
Qtcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2S
Qtcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsω
Ntcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2P
Ntcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/crops?
Btcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Btcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shapeΫ
?tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2A
?tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddings
6tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchNDSpaceToBatchND-tcn_17/residual_block_2/conv1D_0/Pad:output:0Ktcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shape:output:0Htcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ28
6tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND»
6tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????28
6tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims/dim΄
2tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims
ExpandDims?tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND:output:0?tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ24
2tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims
Ctcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpLtcn_17_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02E
Ctcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpΆ
8tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/dim½
4tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1
ExpandDimsKtcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0Atcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@26
4tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1½
'tcn_17/residual_block_2/conv1D_0/conv1dConv2D;tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims:output:0=tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2)
'tcn_17/residual_block_2/conv1D_0/conv1dχ
/tcn_17/residual_block_2/conv1D_0/conv1d/SqueezeSqueeze0tcn_17/residual_block_2/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????21
/tcn_17/residual_block_2/conv1D_0/conv1d/Squeeze?
Btcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Btcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shapeΥ
<tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2>
<tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/crops
6tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceNDBatchToSpaceND8tcn_17/residual_block_2/conv1D_0/conv1d/Squeeze:output:0Ktcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shape:output:0Etcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????28
6tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceNDπ
7tcn_17/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp@tcn_17_residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7tcn_17/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp
(tcn_17/residual_block_2/conv1D_0/BiasAddBiasAdd?tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND:output:0?tcn_17/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2*
(tcn_17/residual_block_2/conv1D_0/BiasAddΝ
+tcn_17/residual_block_2/activation_388/ReluRelu1tcn_17/residual_block_2/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_2/activation_388/ReluΣ
3tcn_17/residual_block_2/spatial_dropout1d_194/ShapeShape9tcn_17/residual_block_2/activation_388/Relu:activations:0*
T0*
_output_shapes
:25
3tcn_17/residual_block_2/spatial_dropout1d_194/ShapeΠ
Atcn_17/residual_block_2/spatial_dropout1d_194/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Atcn_17/residual_block_2/spatial_dropout1d_194/strided_slice/stackΤ
Ctcn_17/residual_block_2/spatial_dropout1d_194/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_2/spatial_dropout1d_194/strided_slice/stack_1Τ
Ctcn_17/residual_block_2/spatial_dropout1d_194/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_2/spatial_dropout1d_194/strided_slice/stack_2φ
;tcn_17/residual_block_2/spatial_dropout1d_194/strided_sliceStridedSlice<tcn_17/residual_block_2/spatial_dropout1d_194/Shape:output:0Jtcn_17/residual_block_2/spatial_dropout1d_194/strided_slice/stack:output:0Ltcn_17/residual_block_2/spatial_dropout1d_194/strided_slice/stack_1:output:0Ltcn_17/residual_block_2/spatial_dropout1d_194/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;tcn_17/residual_block_2/spatial_dropout1d_194/strided_sliceΤ
Ctcn_17/residual_block_2/spatial_dropout1d_194/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_2/spatial_dropout1d_194/strided_slice_1/stackΨ
Etcn_17/residual_block_2/spatial_dropout1d_194/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Etcn_17/residual_block_2/spatial_dropout1d_194/strided_slice_1/stack_1Ψ
Etcn_17/residual_block_2/spatial_dropout1d_194/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Etcn_17/residual_block_2/spatial_dropout1d_194/strided_slice_1/stack_2
=tcn_17/residual_block_2/spatial_dropout1d_194/strided_slice_1StridedSlice<tcn_17/residual_block_2/spatial_dropout1d_194/Shape:output:0Ltcn_17/residual_block_2/spatial_dropout1d_194/strided_slice_1/stack:output:0Ntcn_17/residual_block_2/spatial_dropout1d_194/strided_slice_1/stack_1:output:0Ntcn_17/residual_block_2/spatial_dropout1d_194/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=tcn_17/residual_block_2/spatial_dropout1d_194/strided_slice_1Ώ
;tcn_17/residual_block_2/spatial_dropout1d_194/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2=
;tcn_17/residual_block_2/spatial_dropout1d_194/dropout/ConstΆ
9tcn_17/residual_block_2/spatial_dropout1d_194/dropout/MulMul9tcn_17/residual_block_2/activation_388/Relu:activations:0Dtcn_17/residual_block_2/spatial_dropout1d_194/dropout/Const:output:0*
T0*-
_output_shapes
:?????????2;
9tcn_17/residual_block_2/spatial_dropout1d_194/dropout/Mulή
Ltcn_17/residual_block_2/spatial_dropout1d_194/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2N
Ltcn_17/residual_block_2/spatial_dropout1d_194/dropout/random_uniform/shape/1³
Jtcn_17/residual_block_2/spatial_dropout1d_194/dropout/random_uniform/shapePackDtcn_17/residual_block_2/spatial_dropout1d_194/strided_slice:output:0Utcn_17/residual_block_2/spatial_dropout1d_194/dropout/random_uniform/shape/1:output:0Ftcn_17/residual_block_2/spatial_dropout1d_194/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2L
Jtcn_17/residual_block_2/spatial_dropout1d_194/dropout/random_uniform/shapeΪ
Rtcn_17/residual_block_2/spatial_dropout1d_194/dropout/random_uniform/RandomUniformRandomUniformStcn_17/residual_block_2/spatial_dropout1d_194/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02T
Rtcn_17/residual_block_2/spatial_dropout1d_194/dropout/random_uniform/RandomUniformΡ
Dtcn_17/residual_block_2/spatial_dropout1d_194/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2F
Dtcn_17/residual_block_2/spatial_dropout1d_194/dropout/GreaterEqual/y
Btcn_17/residual_block_2/spatial_dropout1d_194/dropout/GreaterEqualGreaterEqual[tcn_17/residual_block_2/spatial_dropout1d_194/dropout/random_uniform/RandomUniform:output:0Mtcn_17/residual_block_2/spatial_dropout1d_194/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2D
Btcn_17/residual_block_2/spatial_dropout1d_194/dropout/GreaterEqual
:tcn_17/residual_block_2/spatial_dropout1d_194/dropout/CastCastFtcn_17/residual_block_2/spatial_dropout1d_194/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2<
:tcn_17/residual_block_2/spatial_dropout1d_194/dropout/CastΈ
;tcn_17/residual_block_2/spatial_dropout1d_194/dropout/Mul_1Mul=tcn_17/residual_block_2/spatial_dropout1d_194/dropout/Mul:z:0>tcn_17/residual_block_2/spatial_dropout1d_194/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????2=
;tcn_17/residual_block_2/spatial_dropout1d_194/dropout/Mul_1Η
-tcn_17/residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ό               2/
-tcn_17/residual_block_2/conv1D_1/Pad/paddings
$tcn_17/residual_block_2/conv1D_1/PadPad?tcn_17/residual_block_2/spatial_dropout1d_194/dropout/Mul_1:z:06tcn_17/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ό2&
$tcn_17/residual_block_2/conv1D_1/PadΈ
5tcn_17/residual_block_2/conv1D_1/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:27
5tcn_17/residual_block_2/conv1D_1/conv1d/dilation_rateχ
Ttcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ό2V
Ttcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/input_shape
Vtcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2X
Vtcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddings?
Qtcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2S
Qtcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsω
Ntcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2P
Ntcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/crops?
Btcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Btcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shapeΫ
?tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2A
?tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddings
6tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchNDSpaceToBatchND-tcn_17/residual_block_2/conv1D_1/Pad:output:0Ktcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shape:output:0Htcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ28
6tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND»
6tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????28
6tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims/dim΄
2tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims
ExpandDims?tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND:output:0?tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ24
2tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims
Ctcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpLtcn_17_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02E
Ctcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpΆ
8tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/dim½
4tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1
ExpandDimsKtcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Atcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@26
4tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1½
'tcn_17/residual_block_2/conv1D_1/conv1dConv2D;tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims:output:0=tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2)
'tcn_17/residual_block_2/conv1D_1/conv1dχ
/tcn_17/residual_block_2/conv1D_1/conv1d/SqueezeSqueeze0tcn_17/residual_block_2/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????21
/tcn_17/residual_block_2/conv1D_1/conv1d/Squeeze?
Btcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Btcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shapeΥ
<tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2>
<tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/crops
6tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceNDBatchToSpaceND8tcn_17/residual_block_2/conv1D_1/conv1d/Squeeze:output:0Ktcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shape:output:0Etcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????28
6tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceNDπ
7tcn_17/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp@tcn_17_residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7tcn_17/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp
(tcn_17/residual_block_2/conv1D_1/BiasAddBiasAdd?tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND:output:0?tcn_17/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2*
(tcn_17/residual_block_2/conv1D_1/BiasAddΝ
+tcn_17/residual_block_2/activation_389/ReluRelu1tcn_17/residual_block_2/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_2/activation_389/ReluΣ
3tcn_17/residual_block_2/spatial_dropout1d_195/ShapeShape9tcn_17/residual_block_2/activation_389/Relu:activations:0*
T0*
_output_shapes
:25
3tcn_17/residual_block_2/spatial_dropout1d_195/ShapeΠ
Atcn_17/residual_block_2/spatial_dropout1d_195/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Atcn_17/residual_block_2/spatial_dropout1d_195/strided_slice/stackΤ
Ctcn_17/residual_block_2/spatial_dropout1d_195/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_2/spatial_dropout1d_195/strided_slice/stack_1Τ
Ctcn_17/residual_block_2/spatial_dropout1d_195/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_2/spatial_dropout1d_195/strided_slice/stack_2φ
;tcn_17/residual_block_2/spatial_dropout1d_195/strided_sliceStridedSlice<tcn_17/residual_block_2/spatial_dropout1d_195/Shape:output:0Jtcn_17/residual_block_2/spatial_dropout1d_195/strided_slice/stack:output:0Ltcn_17/residual_block_2/spatial_dropout1d_195/strided_slice/stack_1:output:0Ltcn_17/residual_block_2/spatial_dropout1d_195/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;tcn_17/residual_block_2/spatial_dropout1d_195/strided_sliceΤ
Ctcn_17/residual_block_2/spatial_dropout1d_195/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2E
Ctcn_17/residual_block_2/spatial_dropout1d_195/strided_slice_1/stackΨ
Etcn_17/residual_block_2/spatial_dropout1d_195/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Etcn_17/residual_block_2/spatial_dropout1d_195/strided_slice_1/stack_1Ψ
Etcn_17/residual_block_2/spatial_dropout1d_195/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Etcn_17/residual_block_2/spatial_dropout1d_195/strided_slice_1/stack_2
=tcn_17/residual_block_2/spatial_dropout1d_195/strided_slice_1StridedSlice<tcn_17/residual_block_2/spatial_dropout1d_195/Shape:output:0Ltcn_17/residual_block_2/spatial_dropout1d_195/strided_slice_1/stack:output:0Ntcn_17/residual_block_2/spatial_dropout1d_195/strided_slice_1/stack_1:output:0Ntcn_17/residual_block_2/spatial_dropout1d_195/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=tcn_17/residual_block_2/spatial_dropout1d_195/strided_slice_1Ώ
;tcn_17/residual_block_2/spatial_dropout1d_195/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2=
;tcn_17/residual_block_2/spatial_dropout1d_195/dropout/ConstΆ
9tcn_17/residual_block_2/spatial_dropout1d_195/dropout/MulMul9tcn_17/residual_block_2/activation_389/Relu:activations:0Dtcn_17/residual_block_2/spatial_dropout1d_195/dropout/Const:output:0*
T0*-
_output_shapes
:?????????2;
9tcn_17/residual_block_2/spatial_dropout1d_195/dropout/Mulή
Ltcn_17/residual_block_2/spatial_dropout1d_195/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2N
Ltcn_17/residual_block_2/spatial_dropout1d_195/dropout/random_uniform/shape/1³
Jtcn_17/residual_block_2/spatial_dropout1d_195/dropout/random_uniform/shapePackDtcn_17/residual_block_2/spatial_dropout1d_195/strided_slice:output:0Utcn_17/residual_block_2/spatial_dropout1d_195/dropout/random_uniform/shape/1:output:0Ftcn_17/residual_block_2/spatial_dropout1d_195/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2L
Jtcn_17/residual_block_2/spatial_dropout1d_195/dropout/random_uniform/shapeΪ
Rtcn_17/residual_block_2/spatial_dropout1d_195/dropout/random_uniform/RandomUniformRandomUniformStcn_17/residual_block_2/spatial_dropout1d_195/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02T
Rtcn_17/residual_block_2/spatial_dropout1d_195/dropout/random_uniform/RandomUniformΡ
Dtcn_17/residual_block_2/spatial_dropout1d_195/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2F
Dtcn_17/residual_block_2/spatial_dropout1d_195/dropout/GreaterEqual/y
Btcn_17/residual_block_2/spatial_dropout1d_195/dropout/GreaterEqualGreaterEqual[tcn_17/residual_block_2/spatial_dropout1d_195/dropout/random_uniform/RandomUniform:output:0Mtcn_17/residual_block_2/spatial_dropout1d_195/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2D
Btcn_17/residual_block_2/spatial_dropout1d_195/dropout/GreaterEqual
:tcn_17/residual_block_2/spatial_dropout1d_195/dropout/CastCastFtcn_17/residual_block_2/spatial_dropout1d_195/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2<
:tcn_17/residual_block_2/spatial_dropout1d_195/dropout/CastΈ
;tcn_17/residual_block_2/spatial_dropout1d_195/dropout/Mul_1Mul=tcn_17/residual_block_2/spatial_dropout1d_195/dropout/Mul:z:0>tcn_17/residual_block_2/spatial_dropout1d_195/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????2=
;tcn_17/residual_block_2/spatial_dropout1d_195/dropout/Mul_1Ϋ
+tcn_17/residual_block_2/activation_390/ReluRelu?tcn_17/residual_block_2/spatial_dropout1d_195/dropout/Mul_1:z:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_2/activation_390/Reluύ
!tcn_17/residual_block_2/add_2/addAddV29tcn_17/residual_block_1/activation_387/Relu:activations:09tcn_17/residual_block_2/activation_390/Relu:activations:0*
T0*-
_output_shapes
:?????????2#
!tcn_17/residual_block_2/add_2/addΑ
+tcn_17/residual_block_2/activation_391/ReluRelu%tcn_17/residual_block_2/add_2/add:z:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_2/activation_391/ReluΫ
tcn_17/add_3/addAddV29tcn_17/residual_block_0/activation_382/Relu:activations:09tcn_17/residual_block_1/activation_386/Relu:activations:0*
T0*-
_output_shapes
:?????????2
tcn_17/add_3/addΊ
tcn_17/add_3/add_1AddV2tcn_17/add_3/add:z:09tcn_17/residual_block_2/activation_390/Relu:activations:0*
T0*-
_output_shapes
:?????????2
tcn_17/add_3/add_1‘
$tcn_17/lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    2&
$tcn_17/lambda_17/strided_slice/stack₯
&tcn_17/lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2(
&tcn_17/lambda_17/strided_slice/stack_1₯
&tcn_17/lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2(
&tcn_17/lambda_17/strided_slice/stack_2σ
tcn_17/lambda_17/strided_sliceStridedSlicetcn_17/add_3/add_1:z:0-tcn_17/lambda_17/strided_slice/stack:output:0/tcn_17/lambda_17/strided_slice/stack_1:output:0/tcn_17/lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
tcn_17/lambda_17/strided_sliceͺ
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_34/MatMul/ReadVariableOp°
dense_34/MatMulMatMul'tcn_17/lambda_17/strided_slice:output:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_34/MatMul¨
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_34/BiasAdd/ReadVariableOp¦
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_34/BiasAddt
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
dense_34/Reluͺ
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_35/MatMul/ReadVariableOp€
dense_35/MatMulMatMuldense_34/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_35/MatMul¨
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_35/BiasAdd/ReadVariableOp¦
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_35/BiasAddn
IdentityIdentitydense_35/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:?????????:::::::::::::::::::T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs

Ά
C__inference_tcn_17_layer_call_and_return_conditional_losses_1157130

inputsI
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource=
9residual_block_0_conv1d_0_biasadd_readvariableop_resourceI
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource=
9residual_block_0_conv1d_1_biasadd_readvariableop_resourceP
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resourceD
@residual_block_0_matching_conv1d_biasadd_readvariableop_resourceI
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource=
9residual_block_1_conv1d_0_biasadd_readvariableop_resourceI
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource=
9residual_block_1_conv1d_1_biasadd_readvariableop_resourceI
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource=
9residual_block_2_conv1d_0_biasadd_readvariableop_resourceI
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource=
9residual_block_2_conv1d_1_biasadd_readvariableop_resource
identityΉ
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ?               2(
&residual_block_0/conv1D_0/Pad/paddings΅
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:?????????Ώ2
residual_block_0/conv1D_0/Pad­
/residual_block_0/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_0/conv1D_0/conv1d/ExpandDims/dim
+residual_block_0/conv1D_0/conv1d/ExpandDims
ExpandDims&residual_block_0/conv1D_0/Pad:output:08residual_block_0/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Ώ2-
+residual_block_0/conv1D_0/conv1d/ExpandDims
<residual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02>
<residual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_0/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_0/conv1D_0/conv1d/ExpandDims_1/dim 
-residual_block_0/conv1D_0/conv1d/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2/
-residual_block_0/conv1D_0/conv1d/ExpandDims_1‘
 residual_block_0/conv1D_0/conv1dConv2D4residual_block_0/conv1D_0/conv1d/ExpandDims:output:06residual_block_0/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_0/conv1D_0/conv1dβ
(residual_block_0/conv1D_0/conv1d/SqueezeSqueeze)residual_block_0/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_0/conv1D_0/conv1d/SqueezeΫ
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpφ
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/conv1d/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_0/conv1D_0/BiasAddΈ
$residual_block_0/activation_380/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_0/activation_380/ReluΪ
/residual_block_0/spatial_dropout1d_190/IdentityIdentity2residual_block_0/activation_380/Relu:activations:0*
T0*-
_output_shapes
:?????????21
/residual_block_0/spatial_dropout1d_190/IdentityΉ
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ?               2(
&residual_block_0/conv1D_1/Pad/paddingsθ
residual_block_0/conv1D_1/PadPad8residual_block_0/spatial_dropout1d_190/Identity:output:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ2
residual_block_0/conv1D_1/Pad­
/residual_block_0/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_0/conv1D_1/conv1d/ExpandDims/dim
+residual_block_0/conv1D_1/conv1d/ExpandDims
ExpandDims&residual_block_0/conv1D_1/Pad:output:08residual_block_0/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_0/conv1D_1/conv1d/ExpandDims
<residual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_0/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_0/conv1D_1/conv1d/ExpandDims_1/dim‘
-residual_block_0/conv1D_1/conv1d/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_0/conv1D_1/conv1d/ExpandDims_1‘
 residual_block_0/conv1D_1/conv1dConv2D4residual_block_0/conv1D_1/conv1d/ExpandDims:output:06residual_block_0/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_0/conv1D_1/conv1dβ
(residual_block_0/conv1D_1/conv1d/SqueezeSqueeze)residual_block_0/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_0/conv1D_1/conv1d/SqueezeΫ
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpφ
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/conv1d/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_0/conv1D_1/BiasAddΈ
$residual_block_0/activation_381/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_0/activation_381/ReluΪ
/residual_block_0/spatial_dropout1d_191/IdentityIdentity2residual_block_0/activation_381/Relu:activations:0*
T0*-
_output_shapes
:?????????21
/residual_block_0/spatial_dropout1d_191/IdentityΖ
$residual_block_0/activation_382/ReluRelu8residual_block_0/spatial_dropout1d_191/Identity:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_0/activation_382/Relu»
6residual_block_0/matching_conv1D/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????28
6residual_block_0/matching_conv1D/conv1d/ExpandDims/dimϊ
2residual_block_0/matching_conv1D/conv1d/ExpandDims
ExpandDimsinputs?residual_block_0/matching_conv1D/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????24
2residual_block_0/matching_conv1D/conv1d/ExpandDims
Cresidual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpLresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:*
dtype02E
Cresidual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOpΆ
8residual_block_0/matching_conv1D/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8residual_block_0/matching_conv1D/conv1d/ExpandDims_1/dimΌ
4residual_block_0/matching_conv1D/conv1d/ExpandDims_1
ExpandDimsKresidual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOp:value:0Aresidual_block_0/matching_conv1D/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:26
4residual_block_0/matching_conv1D/conv1d/ExpandDims_1Ό
'residual_block_0/matching_conv1D/conv1dConv2D;residual_block_0/matching_conv1D/conv1d/ExpandDims:output:0=residual_block_0/matching_conv1D/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingSAME*
strides
2)
'residual_block_0/matching_conv1D/conv1dχ
/residual_block_0/matching_conv1D/conv1d/SqueezeSqueeze0residual_block_0/matching_conv1D/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????21
/residual_block_0/matching_conv1D/conv1d/Squeezeπ
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/conv1d/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2*
(residual_block_0/matching_conv1D/BiasAddά
residual_block_0/add/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:02residual_block_0/activation_382/Relu:activations:0*
T0*-
_output_shapes
:?????????2
residual_block_0/add/addͺ
$residual_block_0/activation_383/ReluReluresidual_block_0/add/add:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_0/activation_383/ReluΉ
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ~               2(
&residual_block_1/conv1D_0/Pad/paddingsβ
residual_block_1/conv1D_0/PadPad2residual_block_0/activation_383/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ώ2
residual_block_1/conv1D_0/Padͺ
.residual_block_1/conv1D_0/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:20
.residual_block_1/conv1D_0/conv1d/dilation_rateι
Mresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ώ2O
Mresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeϋ
Oresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2Q
Oresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsρ
Jresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsλ
Gresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2I
Gresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/cropsΔ
;residual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shapeΝ
8residual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8residual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddingsέ
/residual_block_1/conv1D_0/conv1d/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_0/Pad:output:0Dresidual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ21
/residual_block_1/conv1D_0/conv1d/SpaceToBatchND­
/residual_block_1/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_1/conv1D_0/conv1d/ExpandDims/dim
+residual_block_1/conv1D_0/conv1d/ExpandDims
ExpandDims8residual_block_1/conv1D_0/conv1d/SpaceToBatchND:output:08residual_block_1/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_1/conv1D_0/conv1d/ExpandDims
<residual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_1/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_1/conv1D_0/conv1d/ExpandDims_1/dim‘
-residual_block_1/conv1D_0/conv1d/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_1/conv1D_0/conv1d/ExpandDims_1‘
 residual_block_1/conv1D_0/conv1dConv2D4residual_block_1/conv1D_0/conv1d/ExpandDims:output:06residual_block_1/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_1/conv1D_0/conv1dβ
(residual_block_1/conv1D_0/conv1d/SqueezeSqueeze)residual_block_1/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_1/conv1D_0/conv1d/SqueezeΔ
;residual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shapeΗ
5residual_block_1/conv1D_0/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        27
5residual_block_1/conv1D_0/conv1d/BatchToSpaceND/cropsε
/residual_block_1/conv1D_0/conv1d/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_0/conv1d/Squeeze:output:0Dresidual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_0/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????21
/residual_block_1/conv1D_0/conv1d/BatchToSpaceNDΫ
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpύ
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/conv1d/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_1/conv1D_0/BiasAddΈ
$residual_block_1/activation_384/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_1/activation_384/ReluΪ
/residual_block_1/spatial_dropout1d_192/IdentityIdentity2residual_block_1/activation_384/Relu:activations:0*
T0*-
_output_shapes
:?????????21
/residual_block_1/spatial_dropout1d_192/IdentityΉ
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ~               2(
&residual_block_1/conv1D_1/Pad/paddingsθ
residual_block_1/conv1D_1/PadPad8residual_block_1/spatial_dropout1d_192/Identity:output:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ώ2
residual_block_1/conv1D_1/Padͺ
.residual_block_1/conv1D_1/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:20
.residual_block_1/conv1D_1/conv1d/dilation_rateι
Mresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ώ2O
Mresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeϋ
Oresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2Q
Oresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsρ
Jresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsλ
Gresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2I
Gresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/cropsΔ
;residual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shapeΝ
8residual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8residual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddingsέ
/residual_block_1/conv1D_1/conv1d/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_1/Pad:output:0Dresidual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ21
/residual_block_1/conv1D_1/conv1d/SpaceToBatchND­
/residual_block_1/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_1/conv1D_1/conv1d/ExpandDims/dim
+residual_block_1/conv1D_1/conv1d/ExpandDims
ExpandDims8residual_block_1/conv1D_1/conv1d/SpaceToBatchND:output:08residual_block_1/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_1/conv1D_1/conv1d/ExpandDims
<residual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_1/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_1/conv1D_1/conv1d/ExpandDims_1/dim‘
-residual_block_1/conv1D_1/conv1d/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_1/conv1D_1/conv1d/ExpandDims_1‘
 residual_block_1/conv1D_1/conv1dConv2D4residual_block_1/conv1D_1/conv1d/ExpandDims:output:06residual_block_1/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_1/conv1D_1/conv1dβ
(residual_block_1/conv1D_1/conv1d/SqueezeSqueeze)residual_block_1/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_1/conv1D_1/conv1d/SqueezeΔ
;residual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shapeΗ
5residual_block_1/conv1D_1/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        27
5residual_block_1/conv1D_1/conv1d/BatchToSpaceND/cropsε
/residual_block_1/conv1D_1/conv1d/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_1/conv1d/Squeeze:output:0Dresidual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_1/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????21
/residual_block_1/conv1D_1/conv1d/BatchToSpaceNDΫ
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpύ
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/conv1d/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_1/conv1D_1/BiasAddΈ
$residual_block_1/activation_385/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_1/activation_385/ReluΪ
/residual_block_1/spatial_dropout1d_193/IdentityIdentity2residual_block_1/activation_385/Relu:activations:0*
T0*-
_output_shapes
:?????????21
/residual_block_1/spatial_dropout1d_193/IdentityΖ
$residual_block_1/activation_386/ReluRelu8residual_block_1/spatial_dropout1d_193/Identity:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_1/activation_386/Reluα
residual_block_1/add_1/addAddV22residual_block_0/activation_383/Relu:activations:02residual_block_1/activation_386/Relu:activations:0*
T0*-
_output_shapes
:?????????2
residual_block_1/add_1/add¬
$residual_block_1/activation_387/ReluReluresidual_block_1/add_1/add:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_1/activation_387/ReluΉ
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ό               2(
&residual_block_2/conv1D_0/Pad/paddingsβ
residual_block_2/conv1D_0/PadPad2residual_block_1/activation_387/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ό2
residual_block_2/conv1D_0/Padͺ
.residual_block_2/conv1D_0/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:20
.residual_block_2/conv1D_0/conv1d/dilation_rateι
Mresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ό2O
Mresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeϋ
Oresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2Q
Oresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsρ
Jresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsλ
Gresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2I
Gresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/cropsΔ
;residual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shapeΝ
8residual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8residual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddingsέ
/residual_block_2/conv1D_0/conv1d/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_0/Pad:output:0Dresidual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ21
/residual_block_2/conv1D_0/conv1d/SpaceToBatchND­
/residual_block_2/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_2/conv1D_0/conv1d/ExpandDims/dim
+residual_block_2/conv1D_0/conv1d/ExpandDims
ExpandDims8residual_block_2/conv1D_0/conv1d/SpaceToBatchND:output:08residual_block_2/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_2/conv1D_0/conv1d/ExpandDims
<residual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_2/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_2/conv1D_0/conv1d/ExpandDims_1/dim‘
-residual_block_2/conv1D_0/conv1d/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_2/conv1D_0/conv1d/ExpandDims_1‘
 residual_block_2/conv1D_0/conv1dConv2D4residual_block_2/conv1D_0/conv1d/ExpandDims:output:06residual_block_2/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_2/conv1D_0/conv1dβ
(residual_block_2/conv1D_0/conv1d/SqueezeSqueeze)residual_block_2/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_2/conv1D_0/conv1d/SqueezeΔ
;residual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shapeΗ
5residual_block_2/conv1D_0/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        27
5residual_block_2/conv1D_0/conv1d/BatchToSpaceND/cropsε
/residual_block_2/conv1D_0/conv1d/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_0/conv1d/Squeeze:output:0Dresidual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_0/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????21
/residual_block_2/conv1D_0/conv1d/BatchToSpaceNDΫ
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpύ
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/conv1d/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_2/conv1D_0/BiasAddΈ
$residual_block_2/activation_388/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_2/activation_388/ReluΪ
/residual_block_2/spatial_dropout1d_194/IdentityIdentity2residual_block_2/activation_388/Relu:activations:0*
T0*-
_output_shapes
:?????????21
/residual_block_2/spatial_dropout1d_194/IdentityΉ
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ό               2(
&residual_block_2/conv1D_1/Pad/paddingsθ
residual_block_2/conv1D_1/PadPad8residual_block_2/spatial_dropout1d_194/Identity:output:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ό2
residual_block_2/conv1D_1/Padͺ
.residual_block_2/conv1D_1/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:20
.residual_block_2/conv1D_1/conv1d/dilation_rateι
Mresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ό2O
Mresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeϋ
Oresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2Q
Oresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsρ
Jresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsλ
Gresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2I
Gresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/cropsΔ
;residual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shapeΝ
8residual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8residual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddingsέ
/residual_block_2/conv1D_1/conv1d/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_1/Pad:output:0Dresidual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ21
/residual_block_2/conv1D_1/conv1d/SpaceToBatchND­
/residual_block_2/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_2/conv1D_1/conv1d/ExpandDims/dim
+residual_block_2/conv1D_1/conv1d/ExpandDims
ExpandDims8residual_block_2/conv1D_1/conv1d/SpaceToBatchND:output:08residual_block_2/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_2/conv1D_1/conv1d/ExpandDims
<residual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_2/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_2/conv1D_1/conv1d/ExpandDims_1/dim‘
-residual_block_2/conv1D_1/conv1d/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_2/conv1D_1/conv1d/ExpandDims_1‘
 residual_block_2/conv1D_1/conv1dConv2D4residual_block_2/conv1D_1/conv1d/ExpandDims:output:06residual_block_2/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_2/conv1D_1/conv1dβ
(residual_block_2/conv1D_1/conv1d/SqueezeSqueeze)residual_block_2/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_2/conv1D_1/conv1d/SqueezeΔ
;residual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shapeΗ
5residual_block_2/conv1D_1/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        27
5residual_block_2/conv1D_1/conv1d/BatchToSpaceND/cropsε
/residual_block_2/conv1D_1/conv1d/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_1/conv1d/Squeeze:output:0Dresidual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_1/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????21
/residual_block_2/conv1D_1/conv1d/BatchToSpaceNDΫ
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpύ
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/conv1d/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_2/conv1D_1/BiasAddΈ
$residual_block_2/activation_389/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_2/activation_389/ReluΪ
/residual_block_2/spatial_dropout1d_195/IdentityIdentity2residual_block_2/activation_389/Relu:activations:0*
T0*-
_output_shapes
:?????????21
/residual_block_2/spatial_dropout1d_195/IdentityΖ
$residual_block_2/activation_390/ReluRelu8residual_block_2/spatial_dropout1d_195/Identity:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_2/activation_390/Reluα
residual_block_2/add_2/addAddV22residual_block_1/activation_387/Relu:activations:02residual_block_2/activation_390/Relu:activations:0*
T0*-
_output_shapes
:?????????2
residual_block_2/add_2/add¬
$residual_block_2/activation_391/ReluReluresidual_block_2/add_2/add:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_2/activation_391/ReluΏ
	add_3/addAddV22residual_block_0/activation_382/Relu:activations:02residual_block_1/activation_386/Relu:activations:0*
T0*-
_output_shapes
:?????????2
	add_3/add
add_3/add_1AddV2add_3/add:z:02residual_block_2/activation_390/Relu:activations:0*
T0*-
_output_shapes
:?????????2
add_3/add_1
lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    2
lambda_17/strided_slice/stack
lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2!
lambda_17/strided_slice/stack_1
lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2!
lambda_17/strided_slice/stack_2Ι
lambda_17/strided_sliceStridedSliceadd_3/add_1:z:0&lambda_17/strided_slice/stack:output:0(lambda_17/strided_slice/stack_1:output:0(lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
lambda_17/strided_sliceu
IdentityIdentity lambda_17/strided_slice:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:?????????:::::::::::::::T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
­
p
R__inference_spatial_dropout1d_195_layer_call_and_return_conditional_losses_1157447

inputs

identity_1p
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????2

Identity

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
q
R__inference_spatial_dropout1d_195_layer_call_and_return_conditional_losses_1155301

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1Ν
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeΠ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2
dropout/GreaterEqual/yΛ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul_1{
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs

S
7__inference_spatial_dropout1d_190_layer_call_fn_1157272

inputs
identityι
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_spatial_dropout1d_190_layer_call_and_return_conditional_losses_11549812
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
q
R__inference_spatial_dropout1d_193_layer_call_and_return_conditional_losses_1157368

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1Ν
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeΠ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2
dropout/GreaterEqual/yΛ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul_1{
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs

S
7__inference_spatial_dropout1d_192_layer_call_fn_1157346

inputs
identityι
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_spatial_dropout1d_192_layer_call_and_return_conditional_losses_11551132
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs

p
7__inference_spatial_dropout1d_193_layer_call_fn_1157378

inputs
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_spatial_dropout1d_193_layer_call_and_return_conditional_losses_11551692
StatefulPartitionedCall€
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
q
R__inference_spatial_dropout1d_195_layer_call_and_return_conditional_losses_1157442

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1Ν
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeΠ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2
dropout/GreaterEqual/yΛ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul_1{
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
q
R__inference_spatial_dropout1d_193_layer_call_and_return_conditional_losses_1155169

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1Ν
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeΠ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2
dropout/GreaterEqual/yΛ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul_1{
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
q
R__inference_spatial_dropout1d_194_layer_call_and_return_conditional_losses_1157405

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1Ν
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeΠ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2
dropout/GreaterEqual/yΛ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul_1{
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
τ
!
 __inference__traced_save_1157663
file_prefix.
*savev2_dense_34_kernel_read_readvariableop,
(savev2_dense_34_bias_read_readvariableop.
*savev2_dense_35_kernel_read_readvariableop,
(savev2_dense_35_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopF
Bsavev2_tcn_17_residual_block_0_conv1d_0_kernel_read_readvariableopD
@savev2_tcn_17_residual_block_0_conv1d_0_bias_read_readvariableopF
Bsavev2_tcn_17_residual_block_0_conv1d_1_kernel_read_readvariableopD
@savev2_tcn_17_residual_block_0_conv1d_1_bias_read_readvariableopM
Isavev2_tcn_17_residual_block_0_matching_conv1d_kernel_read_readvariableopK
Gsavev2_tcn_17_residual_block_0_matching_conv1d_bias_read_readvariableopF
Bsavev2_tcn_17_residual_block_1_conv1d_0_kernel_read_readvariableopD
@savev2_tcn_17_residual_block_1_conv1d_0_bias_read_readvariableopF
Bsavev2_tcn_17_residual_block_1_conv1d_1_kernel_read_readvariableopD
@savev2_tcn_17_residual_block_1_conv1d_1_bias_read_readvariableopF
Bsavev2_tcn_17_residual_block_2_conv1d_0_kernel_read_readvariableopD
@savev2_tcn_17_residual_block_2_conv1d_0_bias_read_readvariableopF
Bsavev2_tcn_17_residual_block_2_conv1d_1_kernel_read_readvariableopD
@savev2_tcn_17_residual_block_2_conv1d_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_34_kernel_m_read_readvariableop3
/savev2_adam_dense_34_bias_m_read_readvariableop5
1savev2_adam_dense_35_kernel_m_read_readvariableop3
/savev2_adam_dense_35_bias_m_read_readvariableopM
Isavev2_adam_tcn_17_residual_block_0_conv1d_0_kernel_m_read_readvariableopK
Gsavev2_adam_tcn_17_residual_block_0_conv1d_0_bias_m_read_readvariableopM
Isavev2_adam_tcn_17_residual_block_0_conv1d_1_kernel_m_read_readvariableopK
Gsavev2_adam_tcn_17_residual_block_0_conv1d_1_bias_m_read_readvariableopT
Psavev2_adam_tcn_17_residual_block_0_matching_conv1d_kernel_m_read_readvariableopR
Nsavev2_adam_tcn_17_residual_block_0_matching_conv1d_bias_m_read_readvariableopM
Isavev2_adam_tcn_17_residual_block_1_conv1d_0_kernel_m_read_readvariableopK
Gsavev2_adam_tcn_17_residual_block_1_conv1d_0_bias_m_read_readvariableopM
Isavev2_adam_tcn_17_residual_block_1_conv1d_1_kernel_m_read_readvariableopK
Gsavev2_adam_tcn_17_residual_block_1_conv1d_1_bias_m_read_readvariableopM
Isavev2_adam_tcn_17_residual_block_2_conv1d_0_kernel_m_read_readvariableopK
Gsavev2_adam_tcn_17_residual_block_2_conv1d_0_bias_m_read_readvariableopM
Isavev2_adam_tcn_17_residual_block_2_conv1d_1_kernel_m_read_readvariableopK
Gsavev2_adam_tcn_17_residual_block_2_conv1d_1_bias_m_read_readvariableop5
1savev2_adam_dense_34_kernel_v_read_readvariableop3
/savev2_adam_dense_34_bias_v_read_readvariableop5
1savev2_adam_dense_35_kernel_v_read_readvariableop3
/savev2_adam_dense_35_bias_v_read_readvariableopM
Isavev2_adam_tcn_17_residual_block_0_conv1d_0_kernel_v_read_readvariableopK
Gsavev2_adam_tcn_17_residual_block_0_conv1d_0_bias_v_read_readvariableopM
Isavev2_adam_tcn_17_residual_block_0_conv1d_1_kernel_v_read_readvariableopK
Gsavev2_adam_tcn_17_residual_block_0_conv1d_1_bias_v_read_readvariableopT
Psavev2_adam_tcn_17_residual_block_0_matching_conv1d_kernel_v_read_readvariableopR
Nsavev2_adam_tcn_17_residual_block_0_matching_conv1d_bias_v_read_readvariableopM
Isavev2_adam_tcn_17_residual_block_1_conv1d_0_kernel_v_read_readvariableopK
Gsavev2_adam_tcn_17_residual_block_1_conv1d_0_bias_v_read_readvariableopM
Isavev2_adam_tcn_17_residual_block_1_conv1d_1_kernel_v_read_readvariableopK
Gsavev2_adam_tcn_17_residual_block_1_conv1d_1_bias_v_read_readvariableopM
Isavev2_adam_tcn_17_residual_block_2_conv1d_0_kernel_v_read_readvariableopK
Gsavev2_adam_tcn_17_residual_block_2_conv1d_0_bias_v_read_readvariableopM
Isavev2_adam_tcn_17_residual_block_2_conv1d_1_kernel_v_read_readvariableopK
Gsavev2_adam_tcn_17_residual_block_2_conv1d_1_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2f9269657307424b8b76af6b9b0350c8/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*€
valueB>B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*
valueB>B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices― 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_34_kernel_read_readvariableop(savev2_dense_34_bias_read_readvariableop*savev2_dense_35_kernel_read_readvariableop(savev2_dense_35_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopBsavev2_tcn_17_residual_block_0_conv1d_0_kernel_read_readvariableop@savev2_tcn_17_residual_block_0_conv1d_0_bias_read_readvariableopBsavev2_tcn_17_residual_block_0_conv1d_1_kernel_read_readvariableop@savev2_tcn_17_residual_block_0_conv1d_1_bias_read_readvariableopIsavev2_tcn_17_residual_block_0_matching_conv1d_kernel_read_readvariableopGsavev2_tcn_17_residual_block_0_matching_conv1d_bias_read_readvariableopBsavev2_tcn_17_residual_block_1_conv1d_0_kernel_read_readvariableop@savev2_tcn_17_residual_block_1_conv1d_0_bias_read_readvariableopBsavev2_tcn_17_residual_block_1_conv1d_1_kernel_read_readvariableop@savev2_tcn_17_residual_block_1_conv1d_1_bias_read_readvariableopBsavev2_tcn_17_residual_block_2_conv1d_0_kernel_read_readvariableop@savev2_tcn_17_residual_block_2_conv1d_0_bias_read_readvariableopBsavev2_tcn_17_residual_block_2_conv1d_1_kernel_read_readvariableop@savev2_tcn_17_residual_block_2_conv1d_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_34_kernel_m_read_readvariableop/savev2_adam_dense_34_bias_m_read_readvariableop1savev2_adam_dense_35_kernel_m_read_readvariableop/savev2_adam_dense_35_bias_m_read_readvariableopIsavev2_adam_tcn_17_residual_block_0_conv1d_0_kernel_m_read_readvariableopGsavev2_adam_tcn_17_residual_block_0_conv1d_0_bias_m_read_readvariableopIsavev2_adam_tcn_17_residual_block_0_conv1d_1_kernel_m_read_readvariableopGsavev2_adam_tcn_17_residual_block_0_conv1d_1_bias_m_read_readvariableopPsavev2_adam_tcn_17_residual_block_0_matching_conv1d_kernel_m_read_readvariableopNsavev2_adam_tcn_17_residual_block_0_matching_conv1d_bias_m_read_readvariableopIsavev2_adam_tcn_17_residual_block_1_conv1d_0_kernel_m_read_readvariableopGsavev2_adam_tcn_17_residual_block_1_conv1d_0_bias_m_read_readvariableopIsavev2_adam_tcn_17_residual_block_1_conv1d_1_kernel_m_read_readvariableopGsavev2_adam_tcn_17_residual_block_1_conv1d_1_bias_m_read_readvariableopIsavev2_adam_tcn_17_residual_block_2_conv1d_0_kernel_m_read_readvariableopGsavev2_adam_tcn_17_residual_block_2_conv1d_0_bias_m_read_readvariableopIsavev2_adam_tcn_17_residual_block_2_conv1d_1_kernel_m_read_readvariableopGsavev2_adam_tcn_17_residual_block_2_conv1d_1_bias_m_read_readvariableop1savev2_adam_dense_34_kernel_v_read_readvariableop/savev2_adam_dense_34_bias_v_read_readvariableop1savev2_adam_dense_35_kernel_v_read_readvariableop/savev2_adam_dense_35_bias_v_read_readvariableopIsavev2_adam_tcn_17_residual_block_0_conv1d_0_kernel_v_read_readvariableopGsavev2_adam_tcn_17_residual_block_0_conv1d_0_bias_v_read_readvariableopIsavev2_adam_tcn_17_residual_block_0_conv1d_1_kernel_v_read_readvariableopGsavev2_adam_tcn_17_residual_block_0_conv1d_1_bias_v_read_readvariableopPsavev2_adam_tcn_17_residual_block_0_matching_conv1d_kernel_v_read_readvariableopNsavev2_adam_tcn_17_residual_block_0_matching_conv1d_bias_v_read_readvariableopIsavev2_adam_tcn_17_residual_block_1_conv1d_0_kernel_v_read_readvariableopGsavev2_adam_tcn_17_residual_block_1_conv1d_0_bias_v_read_readvariableopIsavev2_adam_tcn_17_residual_block_1_conv1d_1_kernel_v_read_readvariableopGsavev2_adam_tcn_17_residual_block_1_conv1d_1_bias_v_read_readvariableopIsavev2_adam_tcn_17_residual_block_2_conv1d_0_kernel_v_read_readvariableopGsavev2_adam_tcn_17_residual_block_2_conv1d_0_bias_v_read_readvariableopIsavev2_adam_tcn_17_residual_block_2_conv1d_1_kernel_v_read_readvariableopGsavev2_adam_tcn_17_residual_block_2_conv1d_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *L
dtypesB
@2>	2
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*φ
_input_shapesδ
α: :
::
:: : : : : :@::@::::@::@::@::@:: : :
::
::@::@::::@::@::@::@::
::
::@::@::::@::@::@::@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :)
%
#
_output_shapes
:@:!

_output_shapes	
::*&
$
_output_shapes
:@:!

_output_shapes	
::)%
#
_output_shapes
::!

_output_shapes	
::*&
$
_output_shapes
:@:!

_output_shapes	
::*&
$
_output_shapes
:@:!

_output_shapes	
::*&
$
_output_shapes
:@:!

_output_shapes	
::*&
$
_output_shapes
:@:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::)%
#
_output_shapes
:@:!

_output_shapes	
::* &
$
_output_shapes
:@:!!

_output_shapes	
::)"%
#
_output_shapes
::!#

_output_shapes	
::*$&
$
_output_shapes
:@:!%

_output_shapes	
::*&&
$
_output_shapes
:@:!'

_output_shapes	
::*(&
$
_output_shapes
:@:!)

_output_shapes	
::**&
$
_output_shapes
:@:!+

_output_shapes	
::&,"
 
_output_shapes
:
:!-

_output_shapes	
::&."
 
_output_shapes
:
:!/

_output_shapes	
::)0%
#
_output_shapes
:@:!1

_output_shapes	
::*2&
$
_output_shapes
:@:!3

_output_shapes	
::)4%
#
_output_shapes
::!5

_output_shapes	
::*6&
$
_output_shapes
:@:!7

_output_shapes	
::*8&
$
_output_shapes
:@:!9

_output_shapes	
::*:&
$
_output_shapes
:@:!;

_output_shapes	
::*<&
$
_output_shapes
:@:!=

_output_shapes	
::>

_output_shapes
: 
Ο
ΐ
J__inference_sequential_17_layer_call_and_return_conditional_losses_1155988

inputs
tcn_17_1155948
tcn_17_1155950
tcn_17_1155952
tcn_17_1155954
tcn_17_1155956
tcn_17_1155958
tcn_17_1155960
tcn_17_1155962
tcn_17_1155964
tcn_17_1155966
tcn_17_1155968
tcn_17_1155970
tcn_17_1155972
tcn_17_1155974
dense_34_1155977
dense_34_1155979
dense_35_1155982
dense_35_1155984
identity’ dense_34/StatefulPartitionedCall’ dense_35/StatefulPartitionedCall’tcn_17/StatefulPartitionedCallι
tcn_17/StatefulPartitionedCallStatefulPartitionedCallinputstcn_17_1155948tcn_17_1155950tcn_17_1155952tcn_17_1155954tcn_17_1155956tcn_17_1155958tcn_17_1155960tcn_17_1155962tcn_17_1155964tcn_17_1155966tcn_17_1155968tcn_17_1155970tcn_17_1155972tcn_17_1155974*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_tcn_17_layer_call_and_return_conditional_losses_11555842 
tcn_17/StatefulPartitionedCallΌ
 dense_34/StatefulPartitionedCallStatefulPartitionedCall'tcn_17/StatefulPartitionedCall:output:0dense_34_1155977dense_34_1155979*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_34_layer_call_and_return_conditional_losses_11558562"
 dense_34/StatefulPartitionedCallΎ
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_1155982dense_35_1155984*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_11558822"
 dense_35/StatefulPartitionedCallε
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall^tcn_17/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:?????????::::::::::::::::::2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2@
tcn_17/StatefulPartitionedCalltcn_17/StatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
₯ν
Ά
C__inference_tcn_17_layer_call_and_return_conditional_losses_1155584

inputsI
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource=
9residual_block_0_conv1d_0_biasadd_readvariableop_resourceI
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource=
9residual_block_0_conv1d_1_biasadd_readvariableop_resourceP
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resourceD
@residual_block_0_matching_conv1d_biasadd_readvariableop_resourceI
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource=
9residual_block_1_conv1d_0_biasadd_readvariableop_resourceI
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource=
9residual_block_1_conv1d_1_biasadd_readvariableop_resourceI
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource=
9residual_block_2_conv1d_0_biasadd_readvariableop_resourceI
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource=
9residual_block_2_conv1d_1_biasadd_readvariableop_resource
identityΉ
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ?               2(
&residual_block_0/conv1D_0/Pad/paddings΅
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:?????????Ώ2
residual_block_0/conv1D_0/Pad­
/residual_block_0/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_0/conv1D_0/conv1d/ExpandDims/dim
+residual_block_0/conv1D_0/conv1d/ExpandDims
ExpandDims&residual_block_0/conv1D_0/Pad:output:08residual_block_0/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Ώ2-
+residual_block_0/conv1D_0/conv1d/ExpandDims
<residual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02>
<residual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_0/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_0/conv1D_0/conv1d/ExpandDims_1/dim 
-residual_block_0/conv1D_0/conv1d/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2/
-residual_block_0/conv1D_0/conv1d/ExpandDims_1‘
 residual_block_0/conv1D_0/conv1dConv2D4residual_block_0/conv1D_0/conv1d/ExpandDims:output:06residual_block_0/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_0/conv1D_0/conv1dβ
(residual_block_0/conv1D_0/conv1d/SqueezeSqueeze)residual_block_0/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_0/conv1D_0/conv1d/SqueezeΫ
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpφ
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/conv1d/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_0/conv1D_0/BiasAddΈ
$residual_block_0/activation_380/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_0/activation_380/ReluΎ
,residual_block_0/spatial_dropout1d_190/ShapeShape2residual_block_0/activation_380/Relu:activations:0*
T0*
_output_shapes
:2.
,residual_block_0/spatial_dropout1d_190/ShapeΒ
:residual_block_0/spatial_dropout1d_190/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:residual_block_0/spatial_dropout1d_190/strided_slice/stackΖ
<residual_block_0/spatial_dropout1d_190/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_0/spatial_dropout1d_190/strided_slice/stack_1Ζ
<residual_block_0/spatial_dropout1d_190/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_0/spatial_dropout1d_190/strided_slice/stack_2Μ
4residual_block_0/spatial_dropout1d_190/strided_sliceStridedSlice5residual_block_0/spatial_dropout1d_190/Shape:output:0Cresidual_block_0/spatial_dropout1d_190/strided_slice/stack:output:0Eresidual_block_0/spatial_dropout1d_190/strided_slice/stack_1:output:0Eresidual_block_0/spatial_dropout1d_190/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4residual_block_0/spatial_dropout1d_190/strided_sliceΖ
<residual_block_0/spatial_dropout1d_190/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_0/spatial_dropout1d_190/strided_slice_1/stackΚ
>residual_block_0/spatial_dropout1d_190/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_0/spatial_dropout1d_190/strided_slice_1/stack_1Κ
>residual_block_0/spatial_dropout1d_190/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_0/spatial_dropout1d_190/strided_slice_1/stack_2Φ
6residual_block_0/spatial_dropout1d_190/strided_slice_1StridedSlice5residual_block_0/spatial_dropout1d_190/Shape:output:0Eresidual_block_0/spatial_dropout1d_190/strided_slice_1/stack:output:0Gresidual_block_0/spatial_dropout1d_190/strided_slice_1/stack_1:output:0Gresidual_block_0/spatial_dropout1d_190/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6residual_block_0/spatial_dropout1d_190/strided_slice_1±
4residual_block_0/spatial_dropout1d_190/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?26
4residual_block_0/spatial_dropout1d_190/dropout/Const
2residual_block_0/spatial_dropout1d_190/dropout/MulMul2residual_block_0/activation_380/Relu:activations:0=residual_block_0/spatial_dropout1d_190/dropout/Const:output:0*
T0*-
_output_shapes
:?????????24
2residual_block_0/spatial_dropout1d_190/dropout/MulΠ
Eresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2G
Eresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/shape/1
Cresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/shapePack=residual_block_0/spatial_dropout1d_190/strided_slice:output:0Nresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/shape/1:output:0?residual_block_0/spatial_dropout1d_190/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2E
Cresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/shapeΕ
Kresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/RandomUniformRandomUniformLresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02M
Kresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/RandomUniformΓ
=residual_block_0/spatial_dropout1d_190/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2?
=residual_block_0/spatial_dropout1d_190/dropout/GreaterEqual/yη
;residual_block_0/spatial_dropout1d_190/dropout/GreaterEqualGreaterEqualTresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/RandomUniform:output:0Fresidual_block_0/spatial_dropout1d_190/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2=
;residual_block_0/spatial_dropout1d_190/dropout/GreaterEqual
3residual_block_0/spatial_dropout1d_190/dropout/CastCast?residual_block_0/spatial_dropout1d_190/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????25
3residual_block_0/spatial_dropout1d_190/dropout/Cast
4residual_block_0/spatial_dropout1d_190/dropout/Mul_1Mul6residual_block_0/spatial_dropout1d_190/dropout/Mul:z:07residual_block_0/spatial_dropout1d_190/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????26
4residual_block_0/spatial_dropout1d_190/dropout/Mul_1Ή
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ?               2(
&residual_block_0/conv1D_1/Pad/paddingsθ
residual_block_0/conv1D_1/PadPad8residual_block_0/spatial_dropout1d_190/dropout/Mul_1:z:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ2
residual_block_0/conv1D_1/Pad­
/residual_block_0/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_0/conv1D_1/conv1d/ExpandDims/dim
+residual_block_0/conv1D_1/conv1d/ExpandDims
ExpandDims&residual_block_0/conv1D_1/Pad:output:08residual_block_0/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_0/conv1D_1/conv1d/ExpandDims
<residual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_0/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_0/conv1D_1/conv1d/ExpandDims_1/dim‘
-residual_block_0/conv1D_1/conv1d/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_0/conv1D_1/conv1d/ExpandDims_1‘
 residual_block_0/conv1D_1/conv1dConv2D4residual_block_0/conv1D_1/conv1d/ExpandDims:output:06residual_block_0/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_0/conv1D_1/conv1dβ
(residual_block_0/conv1D_1/conv1d/SqueezeSqueeze)residual_block_0/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_0/conv1D_1/conv1d/SqueezeΫ
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpφ
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/conv1d/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_0/conv1D_1/BiasAddΈ
$residual_block_0/activation_381/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_0/activation_381/ReluΎ
,residual_block_0/spatial_dropout1d_191/ShapeShape2residual_block_0/activation_381/Relu:activations:0*
T0*
_output_shapes
:2.
,residual_block_0/spatial_dropout1d_191/ShapeΒ
:residual_block_0/spatial_dropout1d_191/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:residual_block_0/spatial_dropout1d_191/strided_slice/stackΖ
<residual_block_0/spatial_dropout1d_191/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_0/spatial_dropout1d_191/strided_slice/stack_1Ζ
<residual_block_0/spatial_dropout1d_191/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_0/spatial_dropout1d_191/strided_slice/stack_2Μ
4residual_block_0/spatial_dropout1d_191/strided_sliceStridedSlice5residual_block_0/spatial_dropout1d_191/Shape:output:0Cresidual_block_0/spatial_dropout1d_191/strided_slice/stack:output:0Eresidual_block_0/spatial_dropout1d_191/strided_slice/stack_1:output:0Eresidual_block_0/spatial_dropout1d_191/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4residual_block_0/spatial_dropout1d_191/strided_sliceΖ
<residual_block_0/spatial_dropout1d_191/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_0/spatial_dropout1d_191/strided_slice_1/stackΚ
>residual_block_0/spatial_dropout1d_191/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_0/spatial_dropout1d_191/strided_slice_1/stack_1Κ
>residual_block_0/spatial_dropout1d_191/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_0/spatial_dropout1d_191/strided_slice_1/stack_2Φ
6residual_block_0/spatial_dropout1d_191/strided_slice_1StridedSlice5residual_block_0/spatial_dropout1d_191/Shape:output:0Eresidual_block_0/spatial_dropout1d_191/strided_slice_1/stack:output:0Gresidual_block_0/spatial_dropout1d_191/strided_slice_1/stack_1:output:0Gresidual_block_0/spatial_dropout1d_191/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6residual_block_0/spatial_dropout1d_191/strided_slice_1±
4residual_block_0/spatial_dropout1d_191/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?26
4residual_block_0/spatial_dropout1d_191/dropout/Const
2residual_block_0/spatial_dropout1d_191/dropout/MulMul2residual_block_0/activation_381/Relu:activations:0=residual_block_0/spatial_dropout1d_191/dropout/Const:output:0*
T0*-
_output_shapes
:?????????24
2residual_block_0/spatial_dropout1d_191/dropout/MulΠ
Eresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2G
Eresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/shape/1
Cresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/shapePack=residual_block_0/spatial_dropout1d_191/strided_slice:output:0Nresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/shape/1:output:0?residual_block_0/spatial_dropout1d_191/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2E
Cresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/shapeΕ
Kresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/RandomUniformRandomUniformLresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02M
Kresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/RandomUniformΓ
=residual_block_0/spatial_dropout1d_191/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2?
=residual_block_0/spatial_dropout1d_191/dropout/GreaterEqual/yη
;residual_block_0/spatial_dropout1d_191/dropout/GreaterEqualGreaterEqualTresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/RandomUniform:output:0Fresidual_block_0/spatial_dropout1d_191/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2=
;residual_block_0/spatial_dropout1d_191/dropout/GreaterEqual
3residual_block_0/spatial_dropout1d_191/dropout/CastCast?residual_block_0/spatial_dropout1d_191/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????25
3residual_block_0/spatial_dropout1d_191/dropout/Cast
4residual_block_0/spatial_dropout1d_191/dropout/Mul_1Mul6residual_block_0/spatial_dropout1d_191/dropout/Mul:z:07residual_block_0/spatial_dropout1d_191/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????26
4residual_block_0/spatial_dropout1d_191/dropout/Mul_1Ζ
$residual_block_0/activation_382/ReluRelu8residual_block_0/spatial_dropout1d_191/dropout/Mul_1:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_0/activation_382/Relu»
6residual_block_0/matching_conv1D/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????28
6residual_block_0/matching_conv1D/conv1d/ExpandDims/dimϊ
2residual_block_0/matching_conv1D/conv1d/ExpandDims
ExpandDimsinputs?residual_block_0/matching_conv1D/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????24
2residual_block_0/matching_conv1D/conv1d/ExpandDims
Cresidual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpLresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:*
dtype02E
Cresidual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOpΆ
8residual_block_0/matching_conv1D/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8residual_block_0/matching_conv1D/conv1d/ExpandDims_1/dimΌ
4residual_block_0/matching_conv1D/conv1d/ExpandDims_1
ExpandDimsKresidual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOp:value:0Aresidual_block_0/matching_conv1D/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:26
4residual_block_0/matching_conv1D/conv1d/ExpandDims_1Ό
'residual_block_0/matching_conv1D/conv1dConv2D;residual_block_0/matching_conv1D/conv1d/ExpandDims:output:0=residual_block_0/matching_conv1D/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingSAME*
strides
2)
'residual_block_0/matching_conv1D/conv1dχ
/residual_block_0/matching_conv1D/conv1d/SqueezeSqueeze0residual_block_0/matching_conv1D/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????21
/residual_block_0/matching_conv1D/conv1d/Squeezeπ
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/conv1d/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2*
(residual_block_0/matching_conv1D/BiasAddά
residual_block_0/add/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:02residual_block_0/activation_382/Relu:activations:0*
T0*-
_output_shapes
:?????????2
residual_block_0/add/addͺ
$residual_block_0/activation_383/ReluReluresidual_block_0/add/add:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_0/activation_383/ReluΉ
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ~               2(
&residual_block_1/conv1D_0/Pad/paddingsβ
residual_block_1/conv1D_0/PadPad2residual_block_0/activation_383/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ώ2
residual_block_1/conv1D_0/Padͺ
.residual_block_1/conv1D_0/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:20
.residual_block_1/conv1D_0/conv1d/dilation_rateι
Mresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ώ2O
Mresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeϋ
Oresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2Q
Oresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsρ
Jresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsλ
Gresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2I
Gresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/cropsΔ
;residual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shapeΝ
8residual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8residual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddingsέ
/residual_block_1/conv1D_0/conv1d/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_0/Pad:output:0Dresidual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ21
/residual_block_1/conv1D_0/conv1d/SpaceToBatchND­
/residual_block_1/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_1/conv1D_0/conv1d/ExpandDims/dim
+residual_block_1/conv1D_0/conv1d/ExpandDims
ExpandDims8residual_block_1/conv1D_0/conv1d/SpaceToBatchND:output:08residual_block_1/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_1/conv1D_0/conv1d/ExpandDims
<residual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_1/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_1/conv1D_0/conv1d/ExpandDims_1/dim‘
-residual_block_1/conv1D_0/conv1d/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_1/conv1D_0/conv1d/ExpandDims_1‘
 residual_block_1/conv1D_0/conv1dConv2D4residual_block_1/conv1D_0/conv1d/ExpandDims:output:06residual_block_1/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_1/conv1D_0/conv1dβ
(residual_block_1/conv1D_0/conv1d/SqueezeSqueeze)residual_block_1/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_1/conv1D_0/conv1d/SqueezeΔ
;residual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shapeΗ
5residual_block_1/conv1D_0/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        27
5residual_block_1/conv1D_0/conv1d/BatchToSpaceND/cropsε
/residual_block_1/conv1D_0/conv1d/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_0/conv1d/Squeeze:output:0Dresidual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_0/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????21
/residual_block_1/conv1D_0/conv1d/BatchToSpaceNDΫ
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpύ
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/conv1d/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_1/conv1D_0/BiasAddΈ
$residual_block_1/activation_384/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_1/activation_384/ReluΎ
,residual_block_1/spatial_dropout1d_192/ShapeShape2residual_block_1/activation_384/Relu:activations:0*
T0*
_output_shapes
:2.
,residual_block_1/spatial_dropout1d_192/ShapeΒ
:residual_block_1/spatial_dropout1d_192/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:residual_block_1/spatial_dropout1d_192/strided_slice/stackΖ
<residual_block_1/spatial_dropout1d_192/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_1/spatial_dropout1d_192/strided_slice/stack_1Ζ
<residual_block_1/spatial_dropout1d_192/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_1/spatial_dropout1d_192/strided_slice/stack_2Μ
4residual_block_1/spatial_dropout1d_192/strided_sliceStridedSlice5residual_block_1/spatial_dropout1d_192/Shape:output:0Cresidual_block_1/spatial_dropout1d_192/strided_slice/stack:output:0Eresidual_block_1/spatial_dropout1d_192/strided_slice/stack_1:output:0Eresidual_block_1/spatial_dropout1d_192/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4residual_block_1/spatial_dropout1d_192/strided_sliceΖ
<residual_block_1/spatial_dropout1d_192/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_1/spatial_dropout1d_192/strided_slice_1/stackΚ
>residual_block_1/spatial_dropout1d_192/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_1/spatial_dropout1d_192/strided_slice_1/stack_1Κ
>residual_block_1/spatial_dropout1d_192/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_1/spatial_dropout1d_192/strided_slice_1/stack_2Φ
6residual_block_1/spatial_dropout1d_192/strided_slice_1StridedSlice5residual_block_1/spatial_dropout1d_192/Shape:output:0Eresidual_block_1/spatial_dropout1d_192/strided_slice_1/stack:output:0Gresidual_block_1/spatial_dropout1d_192/strided_slice_1/stack_1:output:0Gresidual_block_1/spatial_dropout1d_192/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6residual_block_1/spatial_dropout1d_192/strided_slice_1±
4residual_block_1/spatial_dropout1d_192/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?26
4residual_block_1/spatial_dropout1d_192/dropout/Const
2residual_block_1/spatial_dropout1d_192/dropout/MulMul2residual_block_1/activation_384/Relu:activations:0=residual_block_1/spatial_dropout1d_192/dropout/Const:output:0*
T0*-
_output_shapes
:?????????24
2residual_block_1/spatial_dropout1d_192/dropout/MulΠ
Eresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2G
Eresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/shape/1
Cresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/shapePack=residual_block_1/spatial_dropout1d_192/strided_slice:output:0Nresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/shape/1:output:0?residual_block_1/spatial_dropout1d_192/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2E
Cresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/shapeΕ
Kresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/RandomUniformRandomUniformLresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02M
Kresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/RandomUniformΓ
=residual_block_1/spatial_dropout1d_192/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2?
=residual_block_1/spatial_dropout1d_192/dropout/GreaterEqual/yη
;residual_block_1/spatial_dropout1d_192/dropout/GreaterEqualGreaterEqualTresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/RandomUniform:output:0Fresidual_block_1/spatial_dropout1d_192/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2=
;residual_block_1/spatial_dropout1d_192/dropout/GreaterEqual
3residual_block_1/spatial_dropout1d_192/dropout/CastCast?residual_block_1/spatial_dropout1d_192/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????25
3residual_block_1/spatial_dropout1d_192/dropout/Cast
4residual_block_1/spatial_dropout1d_192/dropout/Mul_1Mul6residual_block_1/spatial_dropout1d_192/dropout/Mul:z:07residual_block_1/spatial_dropout1d_192/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????26
4residual_block_1/spatial_dropout1d_192/dropout/Mul_1Ή
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ~               2(
&residual_block_1/conv1D_1/Pad/paddingsθ
residual_block_1/conv1D_1/PadPad8residual_block_1/spatial_dropout1d_192/dropout/Mul_1:z:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ώ2
residual_block_1/conv1D_1/Padͺ
.residual_block_1/conv1D_1/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:20
.residual_block_1/conv1D_1/conv1d/dilation_rateι
Mresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ώ2O
Mresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeϋ
Oresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2Q
Oresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsρ
Jresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsλ
Gresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2I
Gresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/cropsΔ
;residual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shapeΝ
8residual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8residual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddingsέ
/residual_block_1/conv1D_1/conv1d/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_1/Pad:output:0Dresidual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ21
/residual_block_1/conv1D_1/conv1d/SpaceToBatchND­
/residual_block_1/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_1/conv1D_1/conv1d/ExpandDims/dim
+residual_block_1/conv1D_1/conv1d/ExpandDims
ExpandDims8residual_block_1/conv1D_1/conv1d/SpaceToBatchND:output:08residual_block_1/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_1/conv1D_1/conv1d/ExpandDims
<residual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_1/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_1/conv1D_1/conv1d/ExpandDims_1/dim‘
-residual_block_1/conv1D_1/conv1d/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_1/conv1D_1/conv1d/ExpandDims_1‘
 residual_block_1/conv1D_1/conv1dConv2D4residual_block_1/conv1D_1/conv1d/ExpandDims:output:06residual_block_1/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_1/conv1D_1/conv1dβ
(residual_block_1/conv1D_1/conv1d/SqueezeSqueeze)residual_block_1/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_1/conv1D_1/conv1d/SqueezeΔ
;residual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shapeΗ
5residual_block_1/conv1D_1/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        27
5residual_block_1/conv1D_1/conv1d/BatchToSpaceND/cropsε
/residual_block_1/conv1D_1/conv1d/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_1/conv1d/Squeeze:output:0Dresidual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_1/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????21
/residual_block_1/conv1D_1/conv1d/BatchToSpaceNDΫ
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpύ
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/conv1d/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_1/conv1D_1/BiasAddΈ
$residual_block_1/activation_385/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_1/activation_385/ReluΎ
,residual_block_1/spatial_dropout1d_193/ShapeShape2residual_block_1/activation_385/Relu:activations:0*
T0*
_output_shapes
:2.
,residual_block_1/spatial_dropout1d_193/ShapeΒ
:residual_block_1/spatial_dropout1d_193/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:residual_block_1/spatial_dropout1d_193/strided_slice/stackΖ
<residual_block_1/spatial_dropout1d_193/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_1/spatial_dropout1d_193/strided_slice/stack_1Ζ
<residual_block_1/spatial_dropout1d_193/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_1/spatial_dropout1d_193/strided_slice/stack_2Μ
4residual_block_1/spatial_dropout1d_193/strided_sliceStridedSlice5residual_block_1/spatial_dropout1d_193/Shape:output:0Cresidual_block_1/spatial_dropout1d_193/strided_slice/stack:output:0Eresidual_block_1/spatial_dropout1d_193/strided_slice/stack_1:output:0Eresidual_block_1/spatial_dropout1d_193/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4residual_block_1/spatial_dropout1d_193/strided_sliceΖ
<residual_block_1/spatial_dropout1d_193/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_1/spatial_dropout1d_193/strided_slice_1/stackΚ
>residual_block_1/spatial_dropout1d_193/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_1/spatial_dropout1d_193/strided_slice_1/stack_1Κ
>residual_block_1/spatial_dropout1d_193/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_1/spatial_dropout1d_193/strided_slice_1/stack_2Φ
6residual_block_1/spatial_dropout1d_193/strided_slice_1StridedSlice5residual_block_1/spatial_dropout1d_193/Shape:output:0Eresidual_block_1/spatial_dropout1d_193/strided_slice_1/stack:output:0Gresidual_block_1/spatial_dropout1d_193/strided_slice_1/stack_1:output:0Gresidual_block_1/spatial_dropout1d_193/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6residual_block_1/spatial_dropout1d_193/strided_slice_1±
4residual_block_1/spatial_dropout1d_193/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?26
4residual_block_1/spatial_dropout1d_193/dropout/Const
2residual_block_1/spatial_dropout1d_193/dropout/MulMul2residual_block_1/activation_385/Relu:activations:0=residual_block_1/spatial_dropout1d_193/dropout/Const:output:0*
T0*-
_output_shapes
:?????????24
2residual_block_1/spatial_dropout1d_193/dropout/MulΠ
Eresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2G
Eresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/shape/1
Cresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/shapePack=residual_block_1/spatial_dropout1d_193/strided_slice:output:0Nresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/shape/1:output:0?residual_block_1/spatial_dropout1d_193/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2E
Cresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/shapeΕ
Kresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/RandomUniformRandomUniformLresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02M
Kresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/RandomUniformΓ
=residual_block_1/spatial_dropout1d_193/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2?
=residual_block_1/spatial_dropout1d_193/dropout/GreaterEqual/yη
;residual_block_1/spatial_dropout1d_193/dropout/GreaterEqualGreaterEqualTresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/RandomUniform:output:0Fresidual_block_1/spatial_dropout1d_193/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2=
;residual_block_1/spatial_dropout1d_193/dropout/GreaterEqual
3residual_block_1/spatial_dropout1d_193/dropout/CastCast?residual_block_1/spatial_dropout1d_193/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????25
3residual_block_1/spatial_dropout1d_193/dropout/Cast
4residual_block_1/spatial_dropout1d_193/dropout/Mul_1Mul6residual_block_1/spatial_dropout1d_193/dropout/Mul:z:07residual_block_1/spatial_dropout1d_193/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????26
4residual_block_1/spatial_dropout1d_193/dropout/Mul_1Ζ
$residual_block_1/activation_386/ReluRelu8residual_block_1/spatial_dropout1d_193/dropout/Mul_1:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_1/activation_386/Reluα
residual_block_1/add_1/addAddV22residual_block_0/activation_383/Relu:activations:02residual_block_1/activation_386/Relu:activations:0*
T0*-
_output_shapes
:?????????2
residual_block_1/add_1/add¬
$residual_block_1/activation_387/ReluReluresidual_block_1/add_1/add:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_1/activation_387/ReluΉ
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ό               2(
&residual_block_2/conv1D_0/Pad/paddingsβ
residual_block_2/conv1D_0/PadPad2residual_block_1/activation_387/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ό2
residual_block_2/conv1D_0/Padͺ
.residual_block_2/conv1D_0/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:20
.residual_block_2/conv1D_0/conv1d/dilation_rateι
Mresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ό2O
Mresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeϋ
Oresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2Q
Oresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsρ
Jresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsλ
Gresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2I
Gresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/cropsΔ
;residual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shapeΝ
8residual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8residual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddingsέ
/residual_block_2/conv1D_0/conv1d/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_0/Pad:output:0Dresidual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ21
/residual_block_2/conv1D_0/conv1d/SpaceToBatchND­
/residual_block_2/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_2/conv1D_0/conv1d/ExpandDims/dim
+residual_block_2/conv1D_0/conv1d/ExpandDims
ExpandDims8residual_block_2/conv1D_0/conv1d/SpaceToBatchND:output:08residual_block_2/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_2/conv1D_0/conv1d/ExpandDims
<residual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_2/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_2/conv1D_0/conv1d/ExpandDims_1/dim‘
-residual_block_2/conv1D_0/conv1d/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_2/conv1D_0/conv1d/ExpandDims_1‘
 residual_block_2/conv1D_0/conv1dConv2D4residual_block_2/conv1D_0/conv1d/ExpandDims:output:06residual_block_2/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_2/conv1D_0/conv1dβ
(residual_block_2/conv1D_0/conv1d/SqueezeSqueeze)residual_block_2/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_2/conv1D_0/conv1d/SqueezeΔ
;residual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shapeΗ
5residual_block_2/conv1D_0/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        27
5residual_block_2/conv1D_0/conv1d/BatchToSpaceND/cropsε
/residual_block_2/conv1D_0/conv1d/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_0/conv1d/Squeeze:output:0Dresidual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_0/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????21
/residual_block_2/conv1D_0/conv1d/BatchToSpaceNDΫ
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpύ
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/conv1d/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_2/conv1D_0/BiasAddΈ
$residual_block_2/activation_388/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_2/activation_388/ReluΎ
,residual_block_2/spatial_dropout1d_194/ShapeShape2residual_block_2/activation_388/Relu:activations:0*
T0*
_output_shapes
:2.
,residual_block_2/spatial_dropout1d_194/ShapeΒ
:residual_block_2/spatial_dropout1d_194/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:residual_block_2/spatial_dropout1d_194/strided_slice/stackΖ
<residual_block_2/spatial_dropout1d_194/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_2/spatial_dropout1d_194/strided_slice/stack_1Ζ
<residual_block_2/spatial_dropout1d_194/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_2/spatial_dropout1d_194/strided_slice/stack_2Μ
4residual_block_2/spatial_dropout1d_194/strided_sliceStridedSlice5residual_block_2/spatial_dropout1d_194/Shape:output:0Cresidual_block_2/spatial_dropout1d_194/strided_slice/stack:output:0Eresidual_block_2/spatial_dropout1d_194/strided_slice/stack_1:output:0Eresidual_block_2/spatial_dropout1d_194/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4residual_block_2/spatial_dropout1d_194/strided_sliceΖ
<residual_block_2/spatial_dropout1d_194/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_2/spatial_dropout1d_194/strided_slice_1/stackΚ
>residual_block_2/spatial_dropout1d_194/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_2/spatial_dropout1d_194/strided_slice_1/stack_1Κ
>residual_block_2/spatial_dropout1d_194/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_2/spatial_dropout1d_194/strided_slice_1/stack_2Φ
6residual_block_2/spatial_dropout1d_194/strided_slice_1StridedSlice5residual_block_2/spatial_dropout1d_194/Shape:output:0Eresidual_block_2/spatial_dropout1d_194/strided_slice_1/stack:output:0Gresidual_block_2/spatial_dropout1d_194/strided_slice_1/stack_1:output:0Gresidual_block_2/spatial_dropout1d_194/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6residual_block_2/spatial_dropout1d_194/strided_slice_1±
4residual_block_2/spatial_dropout1d_194/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?26
4residual_block_2/spatial_dropout1d_194/dropout/Const
2residual_block_2/spatial_dropout1d_194/dropout/MulMul2residual_block_2/activation_388/Relu:activations:0=residual_block_2/spatial_dropout1d_194/dropout/Const:output:0*
T0*-
_output_shapes
:?????????24
2residual_block_2/spatial_dropout1d_194/dropout/MulΠ
Eresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2G
Eresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/shape/1
Cresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/shapePack=residual_block_2/spatial_dropout1d_194/strided_slice:output:0Nresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/shape/1:output:0?residual_block_2/spatial_dropout1d_194/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2E
Cresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/shapeΕ
Kresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/RandomUniformRandomUniformLresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02M
Kresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/RandomUniformΓ
=residual_block_2/spatial_dropout1d_194/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2?
=residual_block_2/spatial_dropout1d_194/dropout/GreaterEqual/yη
;residual_block_2/spatial_dropout1d_194/dropout/GreaterEqualGreaterEqualTresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/RandomUniform:output:0Fresidual_block_2/spatial_dropout1d_194/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2=
;residual_block_2/spatial_dropout1d_194/dropout/GreaterEqual
3residual_block_2/spatial_dropout1d_194/dropout/CastCast?residual_block_2/spatial_dropout1d_194/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????25
3residual_block_2/spatial_dropout1d_194/dropout/Cast
4residual_block_2/spatial_dropout1d_194/dropout/Mul_1Mul6residual_block_2/spatial_dropout1d_194/dropout/Mul:z:07residual_block_2/spatial_dropout1d_194/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????26
4residual_block_2/spatial_dropout1d_194/dropout/Mul_1Ή
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ό               2(
&residual_block_2/conv1D_1/Pad/paddingsθ
residual_block_2/conv1D_1/PadPad8residual_block_2/spatial_dropout1d_194/dropout/Mul_1:z:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ό2
residual_block_2/conv1D_1/Padͺ
.residual_block_2/conv1D_1/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:20
.residual_block_2/conv1D_1/conv1d/dilation_rateι
Mresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ό2O
Mresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeϋ
Oresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2Q
Oresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsρ
Jresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsλ
Gresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2I
Gresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/cropsΔ
;residual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shapeΝ
8residual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8residual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddingsέ
/residual_block_2/conv1D_1/conv1d/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_1/Pad:output:0Dresidual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ21
/residual_block_2/conv1D_1/conv1d/SpaceToBatchND­
/residual_block_2/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_2/conv1D_1/conv1d/ExpandDims/dim
+residual_block_2/conv1D_1/conv1d/ExpandDims
ExpandDims8residual_block_2/conv1D_1/conv1d/SpaceToBatchND:output:08residual_block_2/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_2/conv1D_1/conv1d/ExpandDims
<residual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_2/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_2/conv1D_1/conv1d/ExpandDims_1/dim‘
-residual_block_2/conv1D_1/conv1d/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_2/conv1D_1/conv1d/ExpandDims_1‘
 residual_block_2/conv1D_1/conv1dConv2D4residual_block_2/conv1D_1/conv1d/ExpandDims:output:06residual_block_2/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_2/conv1D_1/conv1dβ
(residual_block_2/conv1D_1/conv1d/SqueezeSqueeze)residual_block_2/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_2/conv1D_1/conv1d/SqueezeΔ
;residual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shapeΗ
5residual_block_2/conv1D_1/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        27
5residual_block_2/conv1D_1/conv1d/BatchToSpaceND/cropsε
/residual_block_2/conv1D_1/conv1d/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_1/conv1d/Squeeze:output:0Dresidual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_1/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????21
/residual_block_2/conv1D_1/conv1d/BatchToSpaceNDΫ
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpύ
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/conv1d/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_2/conv1D_1/BiasAddΈ
$residual_block_2/activation_389/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_2/activation_389/ReluΎ
,residual_block_2/spatial_dropout1d_195/ShapeShape2residual_block_2/activation_389/Relu:activations:0*
T0*
_output_shapes
:2.
,residual_block_2/spatial_dropout1d_195/ShapeΒ
:residual_block_2/spatial_dropout1d_195/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:residual_block_2/spatial_dropout1d_195/strided_slice/stackΖ
<residual_block_2/spatial_dropout1d_195/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_2/spatial_dropout1d_195/strided_slice/stack_1Ζ
<residual_block_2/spatial_dropout1d_195/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_2/spatial_dropout1d_195/strided_slice/stack_2Μ
4residual_block_2/spatial_dropout1d_195/strided_sliceStridedSlice5residual_block_2/spatial_dropout1d_195/Shape:output:0Cresidual_block_2/spatial_dropout1d_195/strided_slice/stack:output:0Eresidual_block_2/spatial_dropout1d_195/strided_slice/stack_1:output:0Eresidual_block_2/spatial_dropout1d_195/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4residual_block_2/spatial_dropout1d_195/strided_sliceΖ
<residual_block_2/spatial_dropout1d_195/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_2/spatial_dropout1d_195/strided_slice_1/stackΚ
>residual_block_2/spatial_dropout1d_195/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_2/spatial_dropout1d_195/strided_slice_1/stack_1Κ
>residual_block_2/spatial_dropout1d_195/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_2/spatial_dropout1d_195/strided_slice_1/stack_2Φ
6residual_block_2/spatial_dropout1d_195/strided_slice_1StridedSlice5residual_block_2/spatial_dropout1d_195/Shape:output:0Eresidual_block_2/spatial_dropout1d_195/strided_slice_1/stack:output:0Gresidual_block_2/spatial_dropout1d_195/strided_slice_1/stack_1:output:0Gresidual_block_2/spatial_dropout1d_195/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6residual_block_2/spatial_dropout1d_195/strided_slice_1±
4residual_block_2/spatial_dropout1d_195/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?26
4residual_block_2/spatial_dropout1d_195/dropout/Const
2residual_block_2/spatial_dropout1d_195/dropout/MulMul2residual_block_2/activation_389/Relu:activations:0=residual_block_2/spatial_dropout1d_195/dropout/Const:output:0*
T0*-
_output_shapes
:?????????24
2residual_block_2/spatial_dropout1d_195/dropout/MulΠ
Eresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2G
Eresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/shape/1
Cresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/shapePack=residual_block_2/spatial_dropout1d_195/strided_slice:output:0Nresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/shape/1:output:0?residual_block_2/spatial_dropout1d_195/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2E
Cresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/shapeΕ
Kresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/RandomUniformRandomUniformLresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02M
Kresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/RandomUniformΓ
=residual_block_2/spatial_dropout1d_195/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2?
=residual_block_2/spatial_dropout1d_195/dropout/GreaterEqual/yη
;residual_block_2/spatial_dropout1d_195/dropout/GreaterEqualGreaterEqualTresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/RandomUniform:output:0Fresidual_block_2/spatial_dropout1d_195/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2=
;residual_block_2/spatial_dropout1d_195/dropout/GreaterEqual
3residual_block_2/spatial_dropout1d_195/dropout/CastCast?residual_block_2/spatial_dropout1d_195/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????25
3residual_block_2/spatial_dropout1d_195/dropout/Cast
4residual_block_2/spatial_dropout1d_195/dropout/Mul_1Mul6residual_block_2/spatial_dropout1d_195/dropout/Mul:z:07residual_block_2/spatial_dropout1d_195/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????26
4residual_block_2/spatial_dropout1d_195/dropout/Mul_1Ζ
$residual_block_2/activation_390/ReluRelu8residual_block_2/spatial_dropout1d_195/dropout/Mul_1:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_2/activation_390/Reluα
residual_block_2/add_2/addAddV22residual_block_1/activation_387/Relu:activations:02residual_block_2/activation_390/Relu:activations:0*
T0*-
_output_shapes
:?????????2
residual_block_2/add_2/add¬
$residual_block_2/activation_391/ReluReluresidual_block_2/add_2/add:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_2/activation_391/ReluΏ
	add_3/addAddV22residual_block_0/activation_382/Relu:activations:02residual_block_1/activation_386/Relu:activations:0*
T0*-
_output_shapes
:?????????2
	add_3/add
add_3/add_1AddV2add_3/add:z:02residual_block_2/activation_390/Relu:activations:0*
T0*-
_output_shapes
:?????????2
add_3/add_1
lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    2
lambda_17/strided_slice/stack
lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2!
lambda_17/strided_slice/stack_1
lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2!
lambda_17/strided_slice/stack_2Ι
lambda_17/strided_sliceStridedSliceadd_3/add_1:z:0&lambda_17/strided_slice/stack:output:0(lambda_17/strided_slice/stack_1:output:0(lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
lambda_17/strided_sliceu
IdentityIdentity lambda_17/strided_slice:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:?????????:::::::::::::::T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
³
­
E__inference_dense_34_layer_call_and_return_conditional_losses_1155856

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

p
7__inference_spatial_dropout1d_190_layer_call_fn_1157267

inputs
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_spatial_dropout1d_190_layer_call_and_return_conditional_losses_11549712
StatefulPartitionedCall€
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
­
p
R__inference_spatial_dropout1d_194_layer_call_and_return_conditional_losses_1155245

inputs

identity_1p
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????2

Identity

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs

ϋ
/__inference_sequential_17_layer_call_fn_1156700

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity’StatefulPartitionedCallΥ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_11560722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
?
q
R__inference_spatial_dropout1d_191_layer_call_and_return_conditional_losses_1157294

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1Ν
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeΠ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2
dropout/GreaterEqual/yΛ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul_1{
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs

p
7__inference_spatial_dropout1d_195_layer_call_fn_1157452

inputs
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_spatial_dropout1d_195_layer_call_and_return_conditional_losses_11553012
StatefulPartitionedCall€
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
³
­
E__inference_dense_34_layer_call_and_return_conditional_losses_1157207

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

p
7__inference_spatial_dropout1d_192_layer_call_fn_1157341

inputs
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_spatial_dropout1d_192_layer_call_and_return_conditional_losses_11551032
StatefulPartitionedCall€
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
q
R__inference_spatial_dropout1d_191_layer_call_and_return_conditional_losses_1155037

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1Ν
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeΠ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2
dropout/GreaterEqual/yΛ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul_1{
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs

S
7__inference_spatial_dropout1d_191_layer_call_fn_1157309

inputs
identityι
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_spatial_dropout1d_191_layer_call_and_return_conditional_losses_11550472
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
ε

*__inference_dense_34_layer_call_fn_1157216

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallω
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_34_layer_call_and_return_conditional_losses_11558562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
­
p
R__inference_spatial_dropout1d_192_layer_call_and_return_conditional_losses_1155113

inputs

identity_1p
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????2

Identity

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
­
p
R__inference_spatial_dropout1d_193_layer_call_and_return_conditional_losses_1157373

inputs

identity_1p
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????2

Identity

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
π	
΄
(__inference_tcn_17_layer_call_fn_1157196

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_tcn_17_layer_call_and_return_conditional_losses_11557482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:?????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
α
Ζ
J__inference_sequential_17_layer_call_and_return_conditional_losses_1155942
tcn_17_input
tcn_17_1155902
tcn_17_1155904
tcn_17_1155906
tcn_17_1155908
tcn_17_1155910
tcn_17_1155912
tcn_17_1155914
tcn_17_1155916
tcn_17_1155918
tcn_17_1155920
tcn_17_1155922
tcn_17_1155924
tcn_17_1155926
tcn_17_1155928
dense_34_1155931
dense_34_1155933
dense_35_1155936
dense_35_1155938
identity’ dense_34/StatefulPartitionedCall’ dense_35/StatefulPartitionedCall’tcn_17/StatefulPartitionedCallο
tcn_17/StatefulPartitionedCallStatefulPartitionedCalltcn_17_inputtcn_17_1155902tcn_17_1155904tcn_17_1155906tcn_17_1155908tcn_17_1155910tcn_17_1155912tcn_17_1155914tcn_17_1155916tcn_17_1155918tcn_17_1155920tcn_17_1155922tcn_17_1155924tcn_17_1155926tcn_17_1155928*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_tcn_17_layer_call_and_return_conditional_losses_11557482 
tcn_17/StatefulPartitionedCallΌ
 dense_34/StatefulPartitionedCallStatefulPartitionedCall'tcn_17/StatefulPartitionedCall:output:0dense_34_1155931dense_34_1155933*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_34_layer_call_and_return_conditional_losses_11558562"
 dense_34/StatefulPartitionedCallΎ
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_1155936dense_35_1155938*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_11558822"
 dense_35/StatefulPartitionedCallε
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall^tcn_17/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:?????????::::::::::::::::::2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2@
tcn_17/StatefulPartitionedCalltcn_17/StatefulPartitionedCall:Z V
,
_output_shapes
:?????????
&
_user_specified_nametcn_17_input
?
q
R__inference_spatial_dropout1d_192_layer_call_and_return_conditional_losses_1157331

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1Ν
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeΠ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2
dropout/GreaterEqual/yΛ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul_1{
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
­
p
R__inference_spatial_dropout1d_193_layer_call_and_return_conditional_losses_1155179

inputs

identity_1p
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????2

Identity

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs


/__inference_sequential_17_layer_call_fn_1156027
tcn_17_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity’StatefulPartitionedCallΫ
StatefulPartitionedCallStatefulPartitionedCalltcn_17_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_11559882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
,
_output_shapes
:?????????
&
_user_specified_nametcn_17_input
?
q
R__inference_spatial_dropout1d_190_layer_call_and_return_conditional_losses_1154971

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1Ν
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeΠ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2
dropout/GreaterEqual/yΛ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul_1{
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
ε

*__inference_dense_35_layer_call_fn_1157235

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallω
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_11558822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

p
7__inference_spatial_dropout1d_191_layer_call_fn_1157304

inputs
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_spatial_dropout1d_191_layer_call_and_return_conditional_losses_11550372
StatefulPartitionedCall€
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs

Ά
C__inference_tcn_17_layer_call_and_return_conditional_losses_1155748

inputsI
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource=
9residual_block_0_conv1d_0_biasadd_readvariableop_resourceI
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource=
9residual_block_0_conv1d_1_biasadd_readvariableop_resourceP
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resourceD
@residual_block_0_matching_conv1d_biasadd_readvariableop_resourceI
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource=
9residual_block_1_conv1d_0_biasadd_readvariableop_resourceI
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource=
9residual_block_1_conv1d_1_biasadd_readvariableop_resourceI
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource=
9residual_block_2_conv1d_0_biasadd_readvariableop_resourceI
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource=
9residual_block_2_conv1d_1_biasadd_readvariableop_resource
identityΉ
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ?               2(
&residual_block_0/conv1D_0/Pad/paddings΅
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:?????????Ώ2
residual_block_0/conv1D_0/Pad­
/residual_block_0/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_0/conv1D_0/conv1d/ExpandDims/dim
+residual_block_0/conv1D_0/conv1d/ExpandDims
ExpandDims&residual_block_0/conv1D_0/Pad:output:08residual_block_0/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Ώ2-
+residual_block_0/conv1D_0/conv1d/ExpandDims
<residual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02>
<residual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_0/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_0/conv1D_0/conv1d/ExpandDims_1/dim 
-residual_block_0/conv1D_0/conv1d/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2/
-residual_block_0/conv1D_0/conv1d/ExpandDims_1‘
 residual_block_0/conv1D_0/conv1dConv2D4residual_block_0/conv1D_0/conv1d/ExpandDims:output:06residual_block_0/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_0/conv1D_0/conv1dβ
(residual_block_0/conv1D_0/conv1d/SqueezeSqueeze)residual_block_0/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_0/conv1D_0/conv1d/SqueezeΫ
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpφ
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/conv1d/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_0/conv1D_0/BiasAddΈ
$residual_block_0/activation_380/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_0/activation_380/ReluΪ
/residual_block_0/spatial_dropout1d_190/IdentityIdentity2residual_block_0/activation_380/Relu:activations:0*
T0*-
_output_shapes
:?????????21
/residual_block_0/spatial_dropout1d_190/IdentityΉ
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ?               2(
&residual_block_0/conv1D_1/Pad/paddingsθ
residual_block_0/conv1D_1/PadPad8residual_block_0/spatial_dropout1d_190/Identity:output:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ2
residual_block_0/conv1D_1/Pad­
/residual_block_0/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_0/conv1D_1/conv1d/ExpandDims/dim
+residual_block_0/conv1D_1/conv1d/ExpandDims
ExpandDims&residual_block_0/conv1D_1/Pad:output:08residual_block_0/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_0/conv1D_1/conv1d/ExpandDims
<residual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_0/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_0/conv1D_1/conv1d/ExpandDims_1/dim‘
-residual_block_0/conv1D_1/conv1d/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_0/conv1D_1/conv1d/ExpandDims_1‘
 residual_block_0/conv1D_1/conv1dConv2D4residual_block_0/conv1D_1/conv1d/ExpandDims:output:06residual_block_0/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_0/conv1D_1/conv1dβ
(residual_block_0/conv1D_1/conv1d/SqueezeSqueeze)residual_block_0/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_0/conv1D_1/conv1d/SqueezeΫ
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpφ
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/conv1d/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_0/conv1D_1/BiasAddΈ
$residual_block_0/activation_381/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_0/activation_381/ReluΪ
/residual_block_0/spatial_dropout1d_191/IdentityIdentity2residual_block_0/activation_381/Relu:activations:0*
T0*-
_output_shapes
:?????????21
/residual_block_0/spatial_dropout1d_191/IdentityΖ
$residual_block_0/activation_382/ReluRelu8residual_block_0/spatial_dropout1d_191/Identity:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_0/activation_382/Relu»
6residual_block_0/matching_conv1D/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????28
6residual_block_0/matching_conv1D/conv1d/ExpandDims/dimϊ
2residual_block_0/matching_conv1D/conv1d/ExpandDims
ExpandDimsinputs?residual_block_0/matching_conv1D/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????24
2residual_block_0/matching_conv1D/conv1d/ExpandDims
Cresidual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpLresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:*
dtype02E
Cresidual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOpΆ
8residual_block_0/matching_conv1D/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8residual_block_0/matching_conv1D/conv1d/ExpandDims_1/dimΌ
4residual_block_0/matching_conv1D/conv1d/ExpandDims_1
ExpandDimsKresidual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOp:value:0Aresidual_block_0/matching_conv1D/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:26
4residual_block_0/matching_conv1D/conv1d/ExpandDims_1Ό
'residual_block_0/matching_conv1D/conv1dConv2D;residual_block_0/matching_conv1D/conv1d/ExpandDims:output:0=residual_block_0/matching_conv1D/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingSAME*
strides
2)
'residual_block_0/matching_conv1D/conv1dχ
/residual_block_0/matching_conv1D/conv1d/SqueezeSqueeze0residual_block_0/matching_conv1D/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????21
/residual_block_0/matching_conv1D/conv1d/Squeezeπ
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/conv1d/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2*
(residual_block_0/matching_conv1D/BiasAddά
residual_block_0/add/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:02residual_block_0/activation_382/Relu:activations:0*
T0*-
_output_shapes
:?????????2
residual_block_0/add/addͺ
$residual_block_0/activation_383/ReluReluresidual_block_0/add/add:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_0/activation_383/ReluΉ
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ~               2(
&residual_block_1/conv1D_0/Pad/paddingsβ
residual_block_1/conv1D_0/PadPad2residual_block_0/activation_383/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ώ2
residual_block_1/conv1D_0/Padͺ
.residual_block_1/conv1D_0/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:20
.residual_block_1/conv1D_0/conv1d/dilation_rateι
Mresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ώ2O
Mresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeϋ
Oresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2Q
Oresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsρ
Jresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsλ
Gresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2I
Gresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/cropsΔ
;residual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shapeΝ
8residual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8residual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddingsέ
/residual_block_1/conv1D_0/conv1d/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_0/Pad:output:0Dresidual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ21
/residual_block_1/conv1D_0/conv1d/SpaceToBatchND­
/residual_block_1/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_1/conv1D_0/conv1d/ExpandDims/dim
+residual_block_1/conv1D_0/conv1d/ExpandDims
ExpandDims8residual_block_1/conv1D_0/conv1d/SpaceToBatchND:output:08residual_block_1/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_1/conv1D_0/conv1d/ExpandDims
<residual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_1/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_1/conv1D_0/conv1d/ExpandDims_1/dim‘
-residual_block_1/conv1D_0/conv1d/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_1/conv1D_0/conv1d/ExpandDims_1‘
 residual_block_1/conv1D_0/conv1dConv2D4residual_block_1/conv1D_0/conv1d/ExpandDims:output:06residual_block_1/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_1/conv1D_0/conv1dβ
(residual_block_1/conv1D_0/conv1d/SqueezeSqueeze)residual_block_1/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_1/conv1D_0/conv1d/SqueezeΔ
;residual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shapeΗ
5residual_block_1/conv1D_0/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        27
5residual_block_1/conv1D_0/conv1d/BatchToSpaceND/cropsε
/residual_block_1/conv1D_0/conv1d/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_0/conv1d/Squeeze:output:0Dresidual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_0/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????21
/residual_block_1/conv1D_0/conv1d/BatchToSpaceNDΫ
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpύ
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/conv1d/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_1/conv1D_0/BiasAddΈ
$residual_block_1/activation_384/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_1/activation_384/ReluΪ
/residual_block_1/spatial_dropout1d_192/IdentityIdentity2residual_block_1/activation_384/Relu:activations:0*
T0*-
_output_shapes
:?????????21
/residual_block_1/spatial_dropout1d_192/IdentityΉ
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ~               2(
&residual_block_1/conv1D_1/Pad/paddingsθ
residual_block_1/conv1D_1/PadPad8residual_block_1/spatial_dropout1d_192/Identity:output:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ώ2
residual_block_1/conv1D_1/Padͺ
.residual_block_1/conv1D_1/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:20
.residual_block_1/conv1D_1/conv1d/dilation_rateι
Mresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ώ2O
Mresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeϋ
Oresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2Q
Oresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsρ
Jresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsλ
Gresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2I
Gresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/cropsΔ
;residual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shapeΝ
8residual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8residual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddingsέ
/residual_block_1/conv1D_1/conv1d/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_1/Pad:output:0Dresidual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ21
/residual_block_1/conv1D_1/conv1d/SpaceToBatchND­
/residual_block_1/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_1/conv1D_1/conv1d/ExpandDims/dim
+residual_block_1/conv1D_1/conv1d/ExpandDims
ExpandDims8residual_block_1/conv1D_1/conv1d/SpaceToBatchND:output:08residual_block_1/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_1/conv1D_1/conv1d/ExpandDims
<residual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_1/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_1/conv1D_1/conv1d/ExpandDims_1/dim‘
-residual_block_1/conv1D_1/conv1d/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_1/conv1D_1/conv1d/ExpandDims_1‘
 residual_block_1/conv1D_1/conv1dConv2D4residual_block_1/conv1D_1/conv1d/ExpandDims:output:06residual_block_1/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_1/conv1D_1/conv1dβ
(residual_block_1/conv1D_1/conv1d/SqueezeSqueeze)residual_block_1/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_1/conv1D_1/conv1d/SqueezeΔ
;residual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shapeΗ
5residual_block_1/conv1D_1/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        27
5residual_block_1/conv1D_1/conv1d/BatchToSpaceND/cropsε
/residual_block_1/conv1D_1/conv1d/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_1/conv1d/Squeeze:output:0Dresidual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_1/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????21
/residual_block_1/conv1D_1/conv1d/BatchToSpaceNDΫ
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpύ
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/conv1d/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_1/conv1D_1/BiasAddΈ
$residual_block_1/activation_385/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_1/activation_385/ReluΪ
/residual_block_1/spatial_dropout1d_193/IdentityIdentity2residual_block_1/activation_385/Relu:activations:0*
T0*-
_output_shapes
:?????????21
/residual_block_1/spatial_dropout1d_193/IdentityΖ
$residual_block_1/activation_386/ReluRelu8residual_block_1/spatial_dropout1d_193/Identity:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_1/activation_386/Reluα
residual_block_1/add_1/addAddV22residual_block_0/activation_383/Relu:activations:02residual_block_1/activation_386/Relu:activations:0*
T0*-
_output_shapes
:?????????2
residual_block_1/add_1/add¬
$residual_block_1/activation_387/ReluReluresidual_block_1/add_1/add:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_1/activation_387/ReluΉ
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ό               2(
&residual_block_2/conv1D_0/Pad/paddingsβ
residual_block_2/conv1D_0/PadPad2residual_block_1/activation_387/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ό2
residual_block_2/conv1D_0/Padͺ
.residual_block_2/conv1D_0/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:20
.residual_block_2/conv1D_0/conv1d/dilation_rateι
Mresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ό2O
Mresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeϋ
Oresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2Q
Oresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsρ
Jresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsλ
Gresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2I
Gresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/cropsΔ
;residual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shapeΝ
8residual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8residual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddingsέ
/residual_block_2/conv1D_0/conv1d/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_0/Pad:output:0Dresidual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ21
/residual_block_2/conv1D_0/conv1d/SpaceToBatchND­
/residual_block_2/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_2/conv1D_0/conv1d/ExpandDims/dim
+residual_block_2/conv1D_0/conv1d/ExpandDims
ExpandDims8residual_block_2/conv1D_0/conv1d/SpaceToBatchND:output:08residual_block_2/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_2/conv1D_0/conv1d/ExpandDims
<residual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_2/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_2/conv1D_0/conv1d/ExpandDims_1/dim‘
-residual_block_2/conv1D_0/conv1d/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_2/conv1D_0/conv1d/ExpandDims_1‘
 residual_block_2/conv1D_0/conv1dConv2D4residual_block_2/conv1D_0/conv1d/ExpandDims:output:06residual_block_2/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_2/conv1D_0/conv1dβ
(residual_block_2/conv1D_0/conv1d/SqueezeSqueeze)residual_block_2/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_2/conv1D_0/conv1d/SqueezeΔ
;residual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shapeΗ
5residual_block_2/conv1D_0/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        27
5residual_block_2/conv1D_0/conv1d/BatchToSpaceND/cropsε
/residual_block_2/conv1D_0/conv1d/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_0/conv1d/Squeeze:output:0Dresidual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_0/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????21
/residual_block_2/conv1D_0/conv1d/BatchToSpaceNDΫ
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpύ
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/conv1d/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_2/conv1D_0/BiasAddΈ
$residual_block_2/activation_388/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_2/activation_388/ReluΪ
/residual_block_2/spatial_dropout1d_194/IdentityIdentity2residual_block_2/activation_388/Relu:activations:0*
T0*-
_output_shapes
:?????????21
/residual_block_2/spatial_dropout1d_194/IdentityΉ
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ό               2(
&residual_block_2/conv1D_1/Pad/paddingsθ
residual_block_2/conv1D_1/PadPad8residual_block_2/spatial_dropout1d_194/Identity:output:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ό2
residual_block_2/conv1D_1/Padͺ
.residual_block_2/conv1D_1/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:20
.residual_block_2/conv1D_1/conv1d/dilation_rateι
Mresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ό2O
Mresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeϋ
Oresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2Q
Oresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsρ
Jresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsλ
Gresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2I
Gresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/cropsΔ
;residual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shapeΝ
8residual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8residual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddingsέ
/residual_block_2/conv1D_1/conv1d/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_1/Pad:output:0Dresidual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ21
/residual_block_2/conv1D_1/conv1d/SpaceToBatchND­
/residual_block_2/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_2/conv1D_1/conv1d/ExpandDims/dim
+residual_block_2/conv1D_1/conv1d/ExpandDims
ExpandDims8residual_block_2/conv1D_1/conv1d/SpaceToBatchND:output:08residual_block_2/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_2/conv1D_1/conv1d/ExpandDims
<residual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_2/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_2/conv1D_1/conv1d/ExpandDims_1/dim‘
-residual_block_2/conv1D_1/conv1d/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_2/conv1D_1/conv1d/ExpandDims_1‘
 residual_block_2/conv1D_1/conv1dConv2D4residual_block_2/conv1D_1/conv1d/ExpandDims:output:06residual_block_2/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_2/conv1D_1/conv1dβ
(residual_block_2/conv1D_1/conv1d/SqueezeSqueeze)residual_block_2/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_2/conv1D_1/conv1d/SqueezeΔ
;residual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shapeΗ
5residual_block_2/conv1D_1/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        27
5residual_block_2/conv1D_1/conv1d/BatchToSpaceND/cropsε
/residual_block_2/conv1D_1/conv1d/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_1/conv1d/Squeeze:output:0Dresidual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_1/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????21
/residual_block_2/conv1D_1/conv1d/BatchToSpaceNDΫ
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpύ
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/conv1d/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_2/conv1D_1/BiasAddΈ
$residual_block_2/activation_389/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_2/activation_389/ReluΪ
/residual_block_2/spatial_dropout1d_195/IdentityIdentity2residual_block_2/activation_389/Relu:activations:0*
T0*-
_output_shapes
:?????????21
/residual_block_2/spatial_dropout1d_195/IdentityΖ
$residual_block_2/activation_390/ReluRelu8residual_block_2/spatial_dropout1d_195/Identity:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_2/activation_390/Reluα
residual_block_2/add_2/addAddV22residual_block_1/activation_387/Relu:activations:02residual_block_2/activation_390/Relu:activations:0*
T0*-
_output_shapes
:?????????2
residual_block_2/add_2/add¬
$residual_block_2/activation_391/ReluReluresidual_block_2/add_2/add:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_2/activation_391/ReluΏ
	add_3/addAddV22residual_block_0/activation_382/Relu:activations:02residual_block_1/activation_386/Relu:activations:0*
T0*-
_output_shapes
:?????????2
	add_3/add
add_3/add_1AddV2add_3/add:z:02residual_block_2/activation_390/Relu:activations:0*
T0*-
_output_shapes
:?????????2
add_3/add_1
lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    2
lambda_17/strided_slice/stack
lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2!
lambda_17/strided_slice/stack_1
lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2!
lambda_17/strided_slice/stack_2Ι
lambda_17/strided_sliceStridedSliceadd_3/add_1:z:0&lambda_17/strided_slice/stack:output:0(lambda_17/strided_slice/stack_1:output:0(lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
lambda_17/strided_sliceu
IdentityIdentity lambda_17/strided_slice:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:?????????:::::::::::::::T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
­
p
R__inference_spatial_dropout1d_191_layer_call_and_return_conditional_losses_1155047

inputs

identity_1p
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????2

Identity

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
Φ
­
E__inference_dense_35_layer_call_and_return_conditional_losses_1155882

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
τΉ
Υ

J__inference_sequential_17_layer_call_and_return_conditional_losses_1156618

inputsP
Ltcn_17_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resourceD
@tcn_17_residual_block_0_conv1d_0_biasadd_readvariableop_resourceP
Ltcn_17_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resourceD
@tcn_17_residual_block_0_conv1d_1_biasadd_readvariableop_resourceW
Stcn_17_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resourceK
Gtcn_17_residual_block_0_matching_conv1d_biasadd_readvariableop_resourceP
Ltcn_17_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resourceD
@tcn_17_residual_block_1_conv1d_0_biasadd_readvariableop_resourceP
Ltcn_17_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resourceD
@tcn_17_residual_block_1_conv1d_1_biasadd_readvariableop_resourceP
Ltcn_17_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resourceD
@tcn_17_residual_block_2_conv1d_0_biasadd_readvariableop_resourceP
Ltcn_17_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resourceD
@tcn_17_residual_block_2_conv1d_1_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identityΗ
-tcn_17/residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ?               2/
-tcn_17/residual_block_0/conv1D_0/Pad/paddingsΚ
$tcn_17/residual_block_0/conv1D_0/PadPadinputs6tcn_17/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:?????????Ώ2&
$tcn_17/residual_block_0/conv1D_0/Pad»
6tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????28
6tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims/dim‘
2tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims
ExpandDims-tcn_17/residual_block_0/conv1D_0/Pad:output:0?tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Ώ24
2tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims
Ctcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpLtcn_17_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02E
Ctcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpΆ
8tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/dimΌ
4tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1
ExpandDimsKtcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0Atcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@26
4tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1½
'tcn_17/residual_block_0/conv1D_0/conv1dConv2D;tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims:output:0=tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2)
'tcn_17/residual_block_0/conv1D_0/conv1dχ
/tcn_17/residual_block_0/conv1D_0/conv1d/SqueezeSqueeze0tcn_17/residual_block_0/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????21
/tcn_17/residual_block_0/conv1D_0/conv1d/Squeezeπ
7tcn_17/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp@tcn_17_residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7tcn_17/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp
(tcn_17/residual_block_0/conv1D_0/BiasAddBiasAdd8tcn_17/residual_block_0/conv1D_0/conv1d/Squeeze:output:0?tcn_17/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2*
(tcn_17/residual_block_0/conv1D_0/BiasAddΝ
+tcn_17/residual_block_0/activation_380/ReluRelu1tcn_17/residual_block_0/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_0/activation_380/Reluο
6tcn_17/residual_block_0/spatial_dropout1d_190/IdentityIdentity9tcn_17/residual_block_0/activation_380/Relu:activations:0*
T0*-
_output_shapes
:?????????28
6tcn_17/residual_block_0/spatial_dropout1d_190/IdentityΗ
-tcn_17/residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ?               2/
-tcn_17/residual_block_0/conv1D_1/Pad/paddings
$tcn_17/residual_block_0/conv1D_1/PadPad?tcn_17/residual_block_0/spatial_dropout1d_190/Identity:output:06tcn_17/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ2&
$tcn_17/residual_block_0/conv1D_1/Pad»
6tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????28
6tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims/dim’
2tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims
ExpandDims-tcn_17/residual_block_0/conv1D_1/Pad:output:0?tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ24
2tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims
Ctcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpLtcn_17_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02E
Ctcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpΆ
8tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/dim½
4tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1
ExpandDimsKtcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Atcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@26
4tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1½
'tcn_17/residual_block_0/conv1D_1/conv1dConv2D;tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims:output:0=tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2)
'tcn_17/residual_block_0/conv1D_1/conv1dχ
/tcn_17/residual_block_0/conv1D_1/conv1d/SqueezeSqueeze0tcn_17/residual_block_0/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????21
/tcn_17/residual_block_0/conv1D_1/conv1d/Squeezeπ
7tcn_17/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp@tcn_17_residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7tcn_17/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp
(tcn_17/residual_block_0/conv1D_1/BiasAddBiasAdd8tcn_17/residual_block_0/conv1D_1/conv1d/Squeeze:output:0?tcn_17/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2*
(tcn_17/residual_block_0/conv1D_1/BiasAddΝ
+tcn_17/residual_block_0/activation_381/ReluRelu1tcn_17/residual_block_0/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_0/activation_381/Reluο
6tcn_17/residual_block_0/spatial_dropout1d_191/IdentityIdentity9tcn_17/residual_block_0/activation_381/Relu:activations:0*
T0*-
_output_shapes
:?????????28
6tcn_17/residual_block_0/spatial_dropout1d_191/IdentityΫ
+tcn_17/residual_block_0/activation_382/ReluRelu?tcn_17/residual_block_0/spatial_dropout1d_191/Identity:output:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_0/activation_382/ReluΙ
=tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2?
=tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims/dim
9tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims
ExpandDimsinputsFtcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2;
9tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims±
Jtcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpStcn_17_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:*
dtype02L
Jtcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOpΔ
?tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2A
?tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/dimΨ
;tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1
ExpandDimsRtcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOp:value:0Htcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:2=
;tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1Ψ
.tcn_17/residual_block_0/matching_conv1D/conv1dConv2DBtcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims:output:0Dtcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingSAME*
strides
20
.tcn_17/residual_block_0/matching_conv1D/conv1d
6tcn_17/residual_block_0/matching_conv1D/conv1d/SqueezeSqueeze7tcn_17/residual_block_0/matching_conv1D/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????28
6tcn_17/residual_block_0/matching_conv1D/conv1d/Squeeze
>tcn_17/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOpGtcn_17_residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02@
>tcn_17/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp?
/tcn_17/residual_block_0/matching_conv1D/BiasAddBiasAdd?tcn_17/residual_block_0/matching_conv1D/conv1d/Squeeze:output:0Ftcn_17/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????21
/tcn_17/residual_block_0/matching_conv1D/BiasAddψ
tcn_17/residual_block_0/add/addAddV28tcn_17/residual_block_0/matching_conv1D/BiasAdd:output:09tcn_17/residual_block_0/activation_382/Relu:activations:0*
T0*-
_output_shapes
:?????????2!
tcn_17/residual_block_0/add/addΏ
+tcn_17/residual_block_0/activation_383/ReluRelu#tcn_17/residual_block_0/add/add:z:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_0/activation_383/ReluΗ
-tcn_17/residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ~               2/
-tcn_17/residual_block_1/conv1D_0/Pad/paddingsώ
$tcn_17/residual_block_1/conv1D_0/PadPad9tcn_17/residual_block_0/activation_383/Relu:activations:06tcn_17/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ώ2&
$tcn_17/residual_block_1/conv1D_0/PadΈ
5tcn_17/residual_block_1/conv1D_0/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:27
5tcn_17/residual_block_1/conv1D_0/conv1d/dilation_rateχ
Ttcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ώ2V
Ttcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/input_shape
Vtcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2X
Vtcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddings?
Qtcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2S
Qtcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsω
Ntcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2P
Ntcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/crops?
Btcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Btcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shapeΫ
?tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2A
?tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddings
6tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchNDSpaceToBatchND-tcn_17/residual_block_1/conv1D_0/Pad:output:0Ktcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shape:output:0Htcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ28
6tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND»
6tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????28
6tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims/dim΄
2tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims
ExpandDims?tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND:output:0?tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ24
2tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims
Ctcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpLtcn_17_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02E
Ctcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpΆ
8tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/dim½
4tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1
ExpandDimsKtcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0Atcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@26
4tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1½
'tcn_17/residual_block_1/conv1D_0/conv1dConv2D;tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims:output:0=tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2)
'tcn_17/residual_block_1/conv1D_0/conv1dχ
/tcn_17/residual_block_1/conv1D_0/conv1d/SqueezeSqueeze0tcn_17/residual_block_1/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????21
/tcn_17/residual_block_1/conv1D_0/conv1d/Squeeze?
Btcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Btcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shapeΥ
<tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2>
<tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/crops
6tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceNDBatchToSpaceND8tcn_17/residual_block_1/conv1D_0/conv1d/Squeeze:output:0Ktcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shape:output:0Etcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????28
6tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceNDπ
7tcn_17/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp@tcn_17_residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7tcn_17/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp
(tcn_17/residual_block_1/conv1D_0/BiasAddBiasAdd?tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND:output:0?tcn_17/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2*
(tcn_17/residual_block_1/conv1D_0/BiasAddΝ
+tcn_17/residual_block_1/activation_384/ReluRelu1tcn_17/residual_block_1/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_1/activation_384/Reluο
6tcn_17/residual_block_1/spatial_dropout1d_192/IdentityIdentity9tcn_17/residual_block_1/activation_384/Relu:activations:0*
T0*-
_output_shapes
:?????????28
6tcn_17/residual_block_1/spatial_dropout1d_192/IdentityΗ
-tcn_17/residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ~               2/
-tcn_17/residual_block_1/conv1D_1/Pad/paddings
$tcn_17/residual_block_1/conv1D_1/PadPad?tcn_17/residual_block_1/spatial_dropout1d_192/Identity:output:06tcn_17/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ώ2&
$tcn_17/residual_block_1/conv1D_1/PadΈ
5tcn_17/residual_block_1/conv1D_1/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:27
5tcn_17/residual_block_1/conv1D_1/conv1d/dilation_rateχ
Ttcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ώ2V
Ttcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/input_shape
Vtcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2X
Vtcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddings?
Qtcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2S
Qtcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsω
Ntcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2P
Ntcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/crops?
Btcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Btcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shapeΫ
?tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2A
?tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddings
6tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchNDSpaceToBatchND-tcn_17/residual_block_1/conv1D_1/Pad:output:0Ktcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shape:output:0Htcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ28
6tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND»
6tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????28
6tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims/dim΄
2tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims
ExpandDims?tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND:output:0?tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ24
2tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims
Ctcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpLtcn_17_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02E
Ctcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpΆ
8tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/dim½
4tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1
ExpandDimsKtcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Atcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@26
4tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1½
'tcn_17/residual_block_1/conv1D_1/conv1dConv2D;tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims:output:0=tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2)
'tcn_17/residual_block_1/conv1D_1/conv1dχ
/tcn_17/residual_block_1/conv1D_1/conv1d/SqueezeSqueeze0tcn_17/residual_block_1/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????21
/tcn_17/residual_block_1/conv1D_1/conv1d/Squeeze?
Btcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Btcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shapeΥ
<tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2>
<tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/crops
6tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceNDBatchToSpaceND8tcn_17/residual_block_1/conv1D_1/conv1d/Squeeze:output:0Ktcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shape:output:0Etcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????28
6tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceNDπ
7tcn_17/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp@tcn_17_residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7tcn_17/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp
(tcn_17/residual_block_1/conv1D_1/BiasAddBiasAdd?tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND:output:0?tcn_17/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2*
(tcn_17/residual_block_1/conv1D_1/BiasAddΝ
+tcn_17/residual_block_1/activation_385/ReluRelu1tcn_17/residual_block_1/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_1/activation_385/Reluο
6tcn_17/residual_block_1/spatial_dropout1d_193/IdentityIdentity9tcn_17/residual_block_1/activation_385/Relu:activations:0*
T0*-
_output_shapes
:?????????28
6tcn_17/residual_block_1/spatial_dropout1d_193/IdentityΫ
+tcn_17/residual_block_1/activation_386/ReluRelu?tcn_17/residual_block_1/spatial_dropout1d_193/Identity:output:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_1/activation_386/Reluύ
!tcn_17/residual_block_1/add_1/addAddV29tcn_17/residual_block_0/activation_383/Relu:activations:09tcn_17/residual_block_1/activation_386/Relu:activations:0*
T0*-
_output_shapes
:?????????2#
!tcn_17/residual_block_1/add_1/addΑ
+tcn_17/residual_block_1/activation_387/ReluRelu%tcn_17/residual_block_1/add_1/add:z:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_1/activation_387/ReluΗ
-tcn_17/residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ό               2/
-tcn_17/residual_block_2/conv1D_0/Pad/paddingsώ
$tcn_17/residual_block_2/conv1D_0/PadPad9tcn_17/residual_block_1/activation_387/Relu:activations:06tcn_17/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ό2&
$tcn_17/residual_block_2/conv1D_0/PadΈ
5tcn_17/residual_block_2/conv1D_0/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:27
5tcn_17/residual_block_2/conv1D_0/conv1d/dilation_rateχ
Ttcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ό2V
Ttcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/input_shape
Vtcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2X
Vtcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddings?
Qtcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2S
Qtcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsω
Ntcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2P
Ntcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/crops?
Btcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Btcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shapeΫ
?tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2A
?tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddings
6tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchNDSpaceToBatchND-tcn_17/residual_block_2/conv1D_0/Pad:output:0Ktcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shape:output:0Htcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ28
6tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND»
6tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????28
6tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims/dim΄
2tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims
ExpandDims?tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND:output:0?tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ24
2tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims
Ctcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpLtcn_17_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02E
Ctcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpΆ
8tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/dim½
4tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1
ExpandDimsKtcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0Atcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@26
4tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1½
'tcn_17/residual_block_2/conv1D_0/conv1dConv2D;tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims:output:0=tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2)
'tcn_17/residual_block_2/conv1D_0/conv1dχ
/tcn_17/residual_block_2/conv1D_0/conv1d/SqueezeSqueeze0tcn_17/residual_block_2/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????21
/tcn_17/residual_block_2/conv1D_0/conv1d/Squeeze?
Btcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Btcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shapeΥ
<tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2>
<tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/crops
6tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceNDBatchToSpaceND8tcn_17/residual_block_2/conv1D_0/conv1d/Squeeze:output:0Ktcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shape:output:0Etcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????28
6tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceNDπ
7tcn_17/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp@tcn_17_residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7tcn_17/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp
(tcn_17/residual_block_2/conv1D_0/BiasAddBiasAdd?tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND:output:0?tcn_17/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2*
(tcn_17/residual_block_2/conv1D_0/BiasAddΝ
+tcn_17/residual_block_2/activation_388/ReluRelu1tcn_17/residual_block_2/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_2/activation_388/Reluο
6tcn_17/residual_block_2/spatial_dropout1d_194/IdentityIdentity9tcn_17/residual_block_2/activation_388/Relu:activations:0*
T0*-
_output_shapes
:?????????28
6tcn_17/residual_block_2/spatial_dropout1d_194/IdentityΗ
-tcn_17/residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ό               2/
-tcn_17/residual_block_2/conv1D_1/Pad/paddings
$tcn_17/residual_block_2/conv1D_1/PadPad?tcn_17/residual_block_2/spatial_dropout1d_194/Identity:output:06tcn_17/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ό2&
$tcn_17/residual_block_2/conv1D_1/PadΈ
5tcn_17/residual_block_2/conv1D_1/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:27
5tcn_17/residual_block_2/conv1D_1/conv1d/dilation_rateχ
Ttcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ό2V
Ttcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/input_shape
Vtcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2X
Vtcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddings?
Qtcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2S
Qtcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsω
Ntcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2P
Ntcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/crops?
Btcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Btcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shapeΫ
?tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2A
?tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddings
6tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchNDSpaceToBatchND-tcn_17/residual_block_2/conv1D_1/Pad:output:0Ktcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shape:output:0Htcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ28
6tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND»
6tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????28
6tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims/dim΄
2tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims
ExpandDims?tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND:output:0?tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ24
2tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims
Ctcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpLtcn_17_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02E
Ctcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpΆ
8tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/dim½
4tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1
ExpandDimsKtcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Atcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@26
4tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1½
'tcn_17/residual_block_2/conv1D_1/conv1dConv2D;tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims:output:0=tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2)
'tcn_17/residual_block_2/conv1D_1/conv1dχ
/tcn_17/residual_block_2/conv1D_1/conv1d/SqueezeSqueeze0tcn_17/residual_block_2/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????21
/tcn_17/residual_block_2/conv1D_1/conv1d/Squeeze?
Btcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2D
Btcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shapeΥ
<tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2>
<tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/crops
6tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceNDBatchToSpaceND8tcn_17/residual_block_2/conv1D_1/conv1d/Squeeze:output:0Ktcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shape:output:0Etcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????28
6tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceNDπ
7tcn_17/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp@tcn_17_residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7tcn_17/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp
(tcn_17/residual_block_2/conv1D_1/BiasAddBiasAdd?tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND:output:0?tcn_17/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2*
(tcn_17/residual_block_2/conv1D_1/BiasAddΝ
+tcn_17/residual_block_2/activation_389/ReluRelu1tcn_17/residual_block_2/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_2/activation_389/Reluο
6tcn_17/residual_block_2/spatial_dropout1d_195/IdentityIdentity9tcn_17/residual_block_2/activation_389/Relu:activations:0*
T0*-
_output_shapes
:?????????28
6tcn_17/residual_block_2/spatial_dropout1d_195/IdentityΫ
+tcn_17/residual_block_2/activation_390/ReluRelu?tcn_17/residual_block_2/spatial_dropout1d_195/Identity:output:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_2/activation_390/Reluύ
!tcn_17/residual_block_2/add_2/addAddV29tcn_17/residual_block_1/activation_387/Relu:activations:09tcn_17/residual_block_2/activation_390/Relu:activations:0*
T0*-
_output_shapes
:?????????2#
!tcn_17/residual_block_2/add_2/addΑ
+tcn_17/residual_block_2/activation_391/ReluRelu%tcn_17/residual_block_2/add_2/add:z:0*
T0*-
_output_shapes
:?????????2-
+tcn_17/residual_block_2/activation_391/ReluΫ
tcn_17/add_3/addAddV29tcn_17/residual_block_0/activation_382/Relu:activations:09tcn_17/residual_block_1/activation_386/Relu:activations:0*
T0*-
_output_shapes
:?????????2
tcn_17/add_3/addΊ
tcn_17/add_3/add_1AddV2tcn_17/add_3/add:z:09tcn_17/residual_block_2/activation_390/Relu:activations:0*
T0*-
_output_shapes
:?????????2
tcn_17/add_3/add_1‘
$tcn_17/lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    2&
$tcn_17/lambda_17/strided_slice/stack₯
&tcn_17/lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2(
&tcn_17/lambda_17/strided_slice/stack_1₯
&tcn_17/lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2(
&tcn_17/lambda_17/strided_slice/stack_2σ
tcn_17/lambda_17/strided_sliceStridedSlicetcn_17/add_3/add_1:z:0-tcn_17/lambda_17/strided_slice/stack:output:0/tcn_17/lambda_17/strided_slice/stack_1:output:0/tcn_17/lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2 
tcn_17/lambda_17/strided_sliceͺ
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_34/MatMul/ReadVariableOp°
dense_34/MatMulMatMul'tcn_17/lambda_17/strided_slice:output:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_34/MatMul¨
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_34/BiasAdd/ReadVariableOp¦
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_34/BiasAddt
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
dense_34/Reluͺ
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_35/MatMul/ReadVariableOp€
dense_35/MatMulMatMuldense_34/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_35/MatMul¨
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_35/BiasAdd/ReadVariableOp¦
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_35/BiasAddn
IdentityIdentitydense_35/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:?????????:::::::::::::::::::T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ή
ρ'
#__inference__traced_restore_1157856
file_prefix$
 assignvariableop_dense_34_kernel$
 assignvariableop_1_dense_34_bias&
"assignvariableop_2_dense_35_kernel$
 assignvariableop_3_dense_35_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate>
:assignvariableop_9_tcn_17_residual_block_0_conv1d_0_kernel=
9assignvariableop_10_tcn_17_residual_block_0_conv1d_0_bias?
;assignvariableop_11_tcn_17_residual_block_0_conv1d_1_kernel=
9assignvariableop_12_tcn_17_residual_block_0_conv1d_1_biasF
Bassignvariableop_13_tcn_17_residual_block_0_matching_conv1d_kernelD
@assignvariableop_14_tcn_17_residual_block_0_matching_conv1d_bias?
;assignvariableop_15_tcn_17_residual_block_1_conv1d_0_kernel=
9assignvariableop_16_tcn_17_residual_block_1_conv1d_0_bias?
;assignvariableop_17_tcn_17_residual_block_1_conv1d_1_kernel=
9assignvariableop_18_tcn_17_residual_block_1_conv1d_1_bias?
;assignvariableop_19_tcn_17_residual_block_2_conv1d_0_kernel=
9assignvariableop_20_tcn_17_residual_block_2_conv1d_0_bias?
;assignvariableop_21_tcn_17_residual_block_2_conv1d_1_kernel=
9assignvariableop_22_tcn_17_residual_block_2_conv1d_1_bias
assignvariableop_23_total
assignvariableop_24_count.
*assignvariableop_25_adam_dense_34_kernel_m,
(assignvariableop_26_adam_dense_34_bias_m.
*assignvariableop_27_adam_dense_35_kernel_m,
(assignvariableop_28_adam_dense_35_bias_mF
Bassignvariableop_29_adam_tcn_17_residual_block_0_conv1d_0_kernel_mD
@assignvariableop_30_adam_tcn_17_residual_block_0_conv1d_0_bias_mF
Bassignvariableop_31_adam_tcn_17_residual_block_0_conv1d_1_kernel_mD
@assignvariableop_32_adam_tcn_17_residual_block_0_conv1d_1_bias_mM
Iassignvariableop_33_adam_tcn_17_residual_block_0_matching_conv1d_kernel_mK
Gassignvariableop_34_adam_tcn_17_residual_block_0_matching_conv1d_bias_mF
Bassignvariableop_35_adam_tcn_17_residual_block_1_conv1d_0_kernel_mD
@assignvariableop_36_adam_tcn_17_residual_block_1_conv1d_0_bias_mF
Bassignvariableop_37_adam_tcn_17_residual_block_1_conv1d_1_kernel_mD
@assignvariableop_38_adam_tcn_17_residual_block_1_conv1d_1_bias_mF
Bassignvariableop_39_adam_tcn_17_residual_block_2_conv1d_0_kernel_mD
@assignvariableop_40_adam_tcn_17_residual_block_2_conv1d_0_bias_mF
Bassignvariableop_41_adam_tcn_17_residual_block_2_conv1d_1_kernel_mD
@assignvariableop_42_adam_tcn_17_residual_block_2_conv1d_1_bias_m.
*assignvariableop_43_adam_dense_34_kernel_v,
(assignvariableop_44_adam_dense_34_bias_v.
*assignvariableop_45_adam_dense_35_kernel_v,
(assignvariableop_46_adam_dense_35_bias_vF
Bassignvariableop_47_adam_tcn_17_residual_block_0_conv1d_0_kernel_vD
@assignvariableop_48_adam_tcn_17_residual_block_0_conv1d_0_bias_vF
Bassignvariableop_49_adam_tcn_17_residual_block_0_conv1d_1_kernel_vD
@assignvariableop_50_adam_tcn_17_residual_block_0_conv1d_1_bias_vM
Iassignvariableop_51_adam_tcn_17_residual_block_0_matching_conv1d_kernel_vK
Gassignvariableop_52_adam_tcn_17_residual_block_0_matching_conv1d_bias_vF
Bassignvariableop_53_adam_tcn_17_residual_block_1_conv1d_0_kernel_vD
@assignvariableop_54_adam_tcn_17_residual_block_1_conv1d_0_bias_vF
Bassignvariableop_55_adam_tcn_17_residual_block_1_conv1d_1_kernel_vD
@assignvariableop_56_adam_tcn_17_residual_block_1_conv1d_1_bias_vF
Bassignvariableop_57_adam_tcn_17_residual_block_2_conv1d_0_kernel_vD
@assignvariableop_58_adam_tcn_17_residual_block_2_conv1d_0_bias_vF
Bassignvariableop_59_adam_tcn_17_residual_block_2_conv1d_1_kernel_vD
@assignvariableop_60_adam_tcn_17_residual_block_2_conv1d_1_bias_v
identity_62’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_41’AssignVariableOp_42’AssignVariableOp_43’AssignVariableOp_44’AssignVariableOp_45’AssignVariableOp_46’AssignVariableOp_47’AssignVariableOp_48’AssignVariableOp_49’AssignVariableOp_5’AssignVariableOp_50’AssignVariableOp_51’AssignVariableOp_52’AssignVariableOp_53’AssignVariableOp_54’AssignVariableOp_55’AssignVariableOp_56’AssignVariableOp_57’AssignVariableOp_58’AssignVariableOp_59’AssignVariableOp_6’AssignVariableOp_60’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*€
valueB>B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*
valueB>B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesδ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesϋ
ψ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*L
dtypesB
@2>	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_34_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1₯
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_34_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_35_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3₯
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_35_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4‘
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6£
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7’
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ͺ
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ώ
AssignVariableOp_9AssignVariableOp:assignvariableop_9_tcn_17_residual_block_0_conv1d_0_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Α
AssignVariableOp_10AssignVariableOp9assignvariableop_10_tcn_17_residual_block_0_conv1d_0_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Γ
AssignVariableOp_11AssignVariableOp;assignvariableop_11_tcn_17_residual_block_0_conv1d_1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Α
AssignVariableOp_12AssignVariableOp9assignvariableop_12_tcn_17_residual_block_0_conv1d_1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Κ
AssignVariableOp_13AssignVariableOpBassignvariableop_13_tcn_17_residual_block_0_matching_conv1d_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Θ
AssignVariableOp_14AssignVariableOp@assignvariableop_14_tcn_17_residual_block_0_matching_conv1d_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Γ
AssignVariableOp_15AssignVariableOp;assignvariableop_15_tcn_17_residual_block_1_conv1d_0_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Α
AssignVariableOp_16AssignVariableOp9assignvariableop_16_tcn_17_residual_block_1_conv1d_0_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Γ
AssignVariableOp_17AssignVariableOp;assignvariableop_17_tcn_17_residual_block_1_conv1d_1_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Α
AssignVariableOp_18AssignVariableOp9assignvariableop_18_tcn_17_residual_block_1_conv1d_1_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Γ
AssignVariableOp_19AssignVariableOp;assignvariableop_19_tcn_17_residual_block_2_conv1d_0_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Α
AssignVariableOp_20AssignVariableOp9assignvariableop_20_tcn_17_residual_block_2_conv1d_0_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Γ
AssignVariableOp_21AssignVariableOp;assignvariableop_21_tcn_17_residual_block_2_conv1d_1_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Α
AssignVariableOp_22AssignVariableOp9assignvariableop_22_tcn_17_residual_block_2_conv1d_1_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23‘
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24‘
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25²
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_34_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26°
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_34_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27²
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_35_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28°
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_35_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Κ
AssignVariableOp_29AssignVariableOpBassignvariableop_29_adam_tcn_17_residual_block_0_conv1d_0_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Θ
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_tcn_17_residual_block_0_conv1d_0_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Κ
AssignVariableOp_31AssignVariableOpBassignvariableop_31_adam_tcn_17_residual_block_0_conv1d_1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Θ
AssignVariableOp_32AssignVariableOp@assignvariableop_32_adam_tcn_17_residual_block_0_conv1d_1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ρ
AssignVariableOp_33AssignVariableOpIassignvariableop_33_adam_tcn_17_residual_block_0_matching_conv1d_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ο
AssignVariableOp_34AssignVariableOpGassignvariableop_34_adam_tcn_17_residual_block_0_matching_conv1d_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Κ
AssignVariableOp_35AssignVariableOpBassignvariableop_35_adam_tcn_17_residual_block_1_conv1d_0_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Θ
AssignVariableOp_36AssignVariableOp@assignvariableop_36_adam_tcn_17_residual_block_1_conv1d_0_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Κ
AssignVariableOp_37AssignVariableOpBassignvariableop_37_adam_tcn_17_residual_block_1_conv1d_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Θ
AssignVariableOp_38AssignVariableOp@assignvariableop_38_adam_tcn_17_residual_block_1_conv1d_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Κ
AssignVariableOp_39AssignVariableOpBassignvariableop_39_adam_tcn_17_residual_block_2_conv1d_0_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Θ
AssignVariableOp_40AssignVariableOp@assignvariableop_40_adam_tcn_17_residual_block_2_conv1d_0_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Κ
AssignVariableOp_41AssignVariableOpBassignvariableop_41_adam_tcn_17_residual_block_2_conv1d_1_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Θ
AssignVariableOp_42AssignVariableOp@assignvariableop_42_adam_tcn_17_residual_block_2_conv1d_1_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43²
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_34_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44°
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_34_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45²
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_35_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46°
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_35_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Κ
AssignVariableOp_47AssignVariableOpBassignvariableop_47_adam_tcn_17_residual_block_0_conv1d_0_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Θ
AssignVariableOp_48AssignVariableOp@assignvariableop_48_adam_tcn_17_residual_block_0_conv1d_0_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Κ
AssignVariableOp_49AssignVariableOpBassignvariableop_49_adam_tcn_17_residual_block_0_conv1d_1_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Θ
AssignVariableOp_50AssignVariableOp@assignvariableop_50_adam_tcn_17_residual_block_0_conv1d_1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Ρ
AssignVariableOp_51AssignVariableOpIassignvariableop_51_adam_tcn_17_residual_block_0_matching_conv1d_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Ο
AssignVariableOp_52AssignVariableOpGassignvariableop_52_adam_tcn_17_residual_block_0_matching_conv1d_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Κ
AssignVariableOp_53AssignVariableOpBassignvariableop_53_adam_tcn_17_residual_block_1_conv1d_0_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Θ
AssignVariableOp_54AssignVariableOp@assignvariableop_54_adam_tcn_17_residual_block_1_conv1d_0_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55Κ
AssignVariableOp_55AssignVariableOpBassignvariableop_55_adam_tcn_17_residual_block_1_conv1d_1_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Θ
AssignVariableOp_56AssignVariableOp@assignvariableop_56_adam_tcn_17_residual_block_1_conv1d_1_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57Κ
AssignVariableOp_57AssignVariableOpBassignvariableop_57_adam_tcn_17_residual_block_2_conv1d_0_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58Θ
AssignVariableOp_58AssignVariableOp@assignvariableop_58_adam_tcn_17_residual_block_2_conv1d_0_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Κ
AssignVariableOp_59AssignVariableOpBassignvariableop_59_adam_tcn_17_residual_block_2_conv1d_1_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Θ
AssignVariableOp_60AssignVariableOp@assignvariableop_60_adam_tcn_17_residual_block_2_conv1d_1_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_609
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_61Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_61
Identity_62IdentityIdentity_61:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_62"#
identity_62Identity_62:output:0*
_input_shapesω
φ: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Φ
­
E__inference_dense_35_layer_call_and_return_conditional_losses_1157226

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs


/__inference_sequential_17_layer_call_fn_1156111
tcn_17_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity’StatefulPartitionedCallΫ
StatefulPartitionedCallStatefulPartitionedCalltcn_17_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_17_layer_call_and_return_conditional_losses_11560722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:?????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
,
_output_shapes
:?????????
&
_user_specified_nametcn_17_input

S
7__inference_spatial_dropout1d_193_layer_call_fn_1157383

inputs
identityι
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_spatial_dropout1d_193_layer_call_and_return_conditional_losses_11551792
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
α
Ζ
J__inference_sequential_17_layer_call_and_return_conditional_losses_1155899
tcn_17_input
tcn_17_1155816
tcn_17_1155818
tcn_17_1155820
tcn_17_1155822
tcn_17_1155824
tcn_17_1155826
tcn_17_1155828
tcn_17_1155830
tcn_17_1155832
tcn_17_1155834
tcn_17_1155836
tcn_17_1155838
tcn_17_1155840
tcn_17_1155842
dense_34_1155867
dense_34_1155869
dense_35_1155893
dense_35_1155895
identity’ dense_34/StatefulPartitionedCall’ dense_35/StatefulPartitionedCall’tcn_17/StatefulPartitionedCallο
tcn_17/StatefulPartitionedCallStatefulPartitionedCalltcn_17_inputtcn_17_1155816tcn_17_1155818tcn_17_1155820tcn_17_1155822tcn_17_1155824tcn_17_1155826tcn_17_1155828tcn_17_1155830tcn_17_1155832tcn_17_1155834tcn_17_1155836tcn_17_1155838tcn_17_1155840tcn_17_1155842*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_tcn_17_layer_call_and_return_conditional_losses_11555842 
tcn_17/StatefulPartitionedCallΌ
 dense_34/StatefulPartitionedCallStatefulPartitionedCall'tcn_17/StatefulPartitionedCall:output:0dense_34_1155867dense_34_1155869*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_34_layer_call_and_return_conditional_losses_11558562"
 dense_34/StatefulPartitionedCallΎ
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_1155893dense_35_1155895*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_11558822"
 dense_35/StatefulPartitionedCallε
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall^tcn_17/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:?????????::::::::::::::::::2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2@
tcn_17/StatefulPartitionedCalltcn_17/StatefulPartitionedCall:Z V
,
_output_shapes
:?????????
&
_user_specified_nametcn_17_input
?
q
R__inference_spatial_dropout1d_194_layer_call_and_return_conditional_losses_1155235

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1Ν
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeΠ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2
dropout/GreaterEqual/yΛ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul_1{
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
­
p
R__inference_spatial_dropout1d_194_layer_call_and_return_conditional_losses_1157410

inputs

identity_1p
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????2

Identity

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
q
R__inference_spatial_dropout1d_192_layer_call_and_return_conditional_losses_1155103

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1Ν
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shapeΠ
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2
dropout/GreaterEqual/yΛ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
dropout/Mul_1{
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs

p
7__inference_spatial_dropout1d_194_layer_call_fn_1157415

inputs
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_spatial_dropout1d_194_layer_call_and_return_conditional_losses_11552352
StatefulPartitionedCall€
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
Ο
ΐ
J__inference_sequential_17_layer_call_and_return_conditional_losses_1156072

inputs
tcn_17_1156032
tcn_17_1156034
tcn_17_1156036
tcn_17_1156038
tcn_17_1156040
tcn_17_1156042
tcn_17_1156044
tcn_17_1156046
tcn_17_1156048
tcn_17_1156050
tcn_17_1156052
tcn_17_1156054
tcn_17_1156056
tcn_17_1156058
dense_34_1156061
dense_34_1156063
dense_35_1156066
dense_35_1156068
identity’ dense_34/StatefulPartitionedCall’ dense_35/StatefulPartitionedCall’tcn_17/StatefulPartitionedCallι
tcn_17/StatefulPartitionedCallStatefulPartitionedCallinputstcn_17_1156032tcn_17_1156034tcn_17_1156036tcn_17_1156038tcn_17_1156040tcn_17_1156042tcn_17_1156044tcn_17_1156046tcn_17_1156048tcn_17_1156050tcn_17_1156052tcn_17_1156054tcn_17_1156056tcn_17_1156058*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_tcn_17_layer_call_and_return_conditional_losses_11557482 
tcn_17/StatefulPartitionedCallΌ
 dense_34/StatefulPartitionedCallStatefulPartitionedCall'tcn_17/StatefulPartitionedCall:output:0dense_34_1156061dense_34_1156063*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_34_layer_call_and_return_conditional_losses_11558562"
 dense_34/StatefulPartitionedCallΎ
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_1156066dense_35_1156068*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_35_layer_call_and_return_conditional_losses_11558822"
 dense_35/StatefulPartitionedCallε
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall^tcn_17/StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:?????????::::::::::::::::::2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2@
tcn_17/StatefulPartitionedCalltcn_17/StatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
­
p
R__inference_spatial_dropout1d_195_layer_call_and_return_conditional_losses_1155311

inputs

identity_1p
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????2

Identity

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
­
p
R__inference_spatial_dropout1d_192_layer_call_and_return_conditional_losses_1157336

inputs

identity_1p
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????2

Identity

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
­
p
R__inference_spatial_dropout1d_190_layer_call_and_return_conditional_losses_1157262

inputs

identity_1p
IdentityIdentityinputs*
T0*=
_output_shapes+
):'???????????????????????????2

Identity

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
₯ν
Ά
C__inference_tcn_17_layer_call_and_return_conditional_losses_1156966

inputsI
Eresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource=
9residual_block_0_conv1d_0_biasadd_readvariableop_resourceI
Eresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource=
9residual_block_0_conv1d_1_biasadd_readvariableop_resourceP
Lresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resourceD
@residual_block_0_matching_conv1d_biasadd_readvariableop_resourceI
Eresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource=
9residual_block_1_conv1d_0_biasadd_readvariableop_resourceI
Eresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource=
9residual_block_1_conv1d_1_biasadd_readvariableop_resourceI
Eresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource=
9residual_block_2_conv1d_0_biasadd_readvariableop_resourceI
Eresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource=
9residual_block_2_conv1d_1_biasadd_readvariableop_resource
identityΉ
&residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ?               2(
&residual_block_0/conv1D_0/Pad/paddings΅
residual_block_0/conv1D_0/PadPadinputs/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:?????????Ώ2
residual_block_0/conv1D_0/Pad­
/residual_block_0/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_0/conv1D_0/conv1d/ExpandDims/dim
+residual_block_0/conv1D_0/conv1d/ExpandDims
ExpandDims&residual_block_0/conv1D_0/Pad:output:08residual_block_0/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Ώ2-
+residual_block_0/conv1D_0/conv1d/ExpandDims
<residual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02>
<residual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_0/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_0/conv1D_0/conv1d/ExpandDims_1/dim 
-residual_block_0/conv1D_0/conv1d/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2/
-residual_block_0/conv1D_0/conv1d/ExpandDims_1‘
 residual_block_0/conv1D_0/conv1dConv2D4residual_block_0/conv1D_0/conv1d/ExpandDims:output:06residual_block_0/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_0/conv1D_0/conv1dβ
(residual_block_0/conv1D_0/conv1d/SqueezeSqueeze)residual_block_0/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_0/conv1D_0/conv1d/SqueezeΫ
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_0/conv1D_0/BiasAdd/ReadVariableOpφ
!residual_block_0/conv1D_0/BiasAddBiasAdd1residual_block_0/conv1D_0/conv1d/Squeeze:output:08residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_0/conv1D_0/BiasAddΈ
$residual_block_0/activation_380/ReluRelu*residual_block_0/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_0/activation_380/ReluΎ
,residual_block_0/spatial_dropout1d_190/ShapeShape2residual_block_0/activation_380/Relu:activations:0*
T0*
_output_shapes
:2.
,residual_block_0/spatial_dropout1d_190/ShapeΒ
:residual_block_0/spatial_dropout1d_190/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:residual_block_0/spatial_dropout1d_190/strided_slice/stackΖ
<residual_block_0/spatial_dropout1d_190/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_0/spatial_dropout1d_190/strided_slice/stack_1Ζ
<residual_block_0/spatial_dropout1d_190/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_0/spatial_dropout1d_190/strided_slice/stack_2Μ
4residual_block_0/spatial_dropout1d_190/strided_sliceStridedSlice5residual_block_0/spatial_dropout1d_190/Shape:output:0Cresidual_block_0/spatial_dropout1d_190/strided_slice/stack:output:0Eresidual_block_0/spatial_dropout1d_190/strided_slice/stack_1:output:0Eresidual_block_0/spatial_dropout1d_190/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4residual_block_0/spatial_dropout1d_190/strided_sliceΖ
<residual_block_0/spatial_dropout1d_190/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_0/spatial_dropout1d_190/strided_slice_1/stackΚ
>residual_block_0/spatial_dropout1d_190/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_0/spatial_dropout1d_190/strided_slice_1/stack_1Κ
>residual_block_0/spatial_dropout1d_190/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_0/spatial_dropout1d_190/strided_slice_1/stack_2Φ
6residual_block_0/spatial_dropout1d_190/strided_slice_1StridedSlice5residual_block_0/spatial_dropout1d_190/Shape:output:0Eresidual_block_0/spatial_dropout1d_190/strided_slice_1/stack:output:0Gresidual_block_0/spatial_dropout1d_190/strided_slice_1/stack_1:output:0Gresidual_block_0/spatial_dropout1d_190/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6residual_block_0/spatial_dropout1d_190/strided_slice_1±
4residual_block_0/spatial_dropout1d_190/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?26
4residual_block_0/spatial_dropout1d_190/dropout/Const
2residual_block_0/spatial_dropout1d_190/dropout/MulMul2residual_block_0/activation_380/Relu:activations:0=residual_block_0/spatial_dropout1d_190/dropout/Const:output:0*
T0*-
_output_shapes
:?????????24
2residual_block_0/spatial_dropout1d_190/dropout/MulΠ
Eresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2G
Eresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/shape/1
Cresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/shapePack=residual_block_0/spatial_dropout1d_190/strided_slice:output:0Nresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/shape/1:output:0?residual_block_0/spatial_dropout1d_190/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2E
Cresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/shapeΕ
Kresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/RandomUniformRandomUniformLresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02M
Kresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/RandomUniformΓ
=residual_block_0/spatial_dropout1d_190/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2?
=residual_block_0/spatial_dropout1d_190/dropout/GreaterEqual/yη
;residual_block_0/spatial_dropout1d_190/dropout/GreaterEqualGreaterEqualTresidual_block_0/spatial_dropout1d_190/dropout/random_uniform/RandomUniform:output:0Fresidual_block_0/spatial_dropout1d_190/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2=
;residual_block_0/spatial_dropout1d_190/dropout/GreaterEqual
3residual_block_0/spatial_dropout1d_190/dropout/CastCast?residual_block_0/spatial_dropout1d_190/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????25
3residual_block_0/spatial_dropout1d_190/dropout/Cast
4residual_block_0/spatial_dropout1d_190/dropout/Mul_1Mul6residual_block_0/spatial_dropout1d_190/dropout/Mul:z:07residual_block_0/spatial_dropout1d_190/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????26
4residual_block_0/spatial_dropout1d_190/dropout/Mul_1Ή
&residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ?               2(
&residual_block_0/conv1D_1/Pad/paddingsθ
residual_block_0/conv1D_1/PadPad8residual_block_0/spatial_dropout1d_190/dropout/Mul_1:z:0/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ2
residual_block_0/conv1D_1/Pad­
/residual_block_0/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_0/conv1D_1/conv1d/ExpandDims/dim
+residual_block_0/conv1D_1/conv1d/ExpandDims
ExpandDims&residual_block_0/conv1D_1/Pad:output:08residual_block_0/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_0/conv1D_1/conv1d/ExpandDims
<residual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_0/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_0/conv1D_1/conv1d/ExpandDims_1/dim‘
-residual_block_0/conv1D_1/conv1d/ExpandDims_1
ExpandDimsDresidual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_0/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_0/conv1D_1/conv1d/ExpandDims_1‘
 residual_block_0/conv1D_1/conv1dConv2D4residual_block_0/conv1D_1/conv1d/ExpandDims:output:06residual_block_0/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_0/conv1D_1/conv1dβ
(residual_block_0/conv1D_1/conv1d/SqueezeSqueeze)residual_block_0/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_0/conv1D_1/conv1d/SqueezeΫ
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_0/conv1D_1/BiasAdd/ReadVariableOpφ
!residual_block_0/conv1D_1/BiasAddBiasAdd1residual_block_0/conv1D_1/conv1d/Squeeze:output:08residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_0/conv1D_1/BiasAddΈ
$residual_block_0/activation_381/ReluRelu*residual_block_0/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_0/activation_381/ReluΎ
,residual_block_0/spatial_dropout1d_191/ShapeShape2residual_block_0/activation_381/Relu:activations:0*
T0*
_output_shapes
:2.
,residual_block_0/spatial_dropout1d_191/ShapeΒ
:residual_block_0/spatial_dropout1d_191/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:residual_block_0/spatial_dropout1d_191/strided_slice/stackΖ
<residual_block_0/spatial_dropout1d_191/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_0/spatial_dropout1d_191/strided_slice/stack_1Ζ
<residual_block_0/spatial_dropout1d_191/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_0/spatial_dropout1d_191/strided_slice/stack_2Μ
4residual_block_0/spatial_dropout1d_191/strided_sliceStridedSlice5residual_block_0/spatial_dropout1d_191/Shape:output:0Cresidual_block_0/spatial_dropout1d_191/strided_slice/stack:output:0Eresidual_block_0/spatial_dropout1d_191/strided_slice/stack_1:output:0Eresidual_block_0/spatial_dropout1d_191/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4residual_block_0/spatial_dropout1d_191/strided_sliceΖ
<residual_block_0/spatial_dropout1d_191/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_0/spatial_dropout1d_191/strided_slice_1/stackΚ
>residual_block_0/spatial_dropout1d_191/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_0/spatial_dropout1d_191/strided_slice_1/stack_1Κ
>residual_block_0/spatial_dropout1d_191/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_0/spatial_dropout1d_191/strided_slice_1/stack_2Φ
6residual_block_0/spatial_dropout1d_191/strided_slice_1StridedSlice5residual_block_0/spatial_dropout1d_191/Shape:output:0Eresidual_block_0/spatial_dropout1d_191/strided_slice_1/stack:output:0Gresidual_block_0/spatial_dropout1d_191/strided_slice_1/stack_1:output:0Gresidual_block_0/spatial_dropout1d_191/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6residual_block_0/spatial_dropout1d_191/strided_slice_1±
4residual_block_0/spatial_dropout1d_191/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?26
4residual_block_0/spatial_dropout1d_191/dropout/Const
2residual_block_0/spatial_dropout1d_191/dropout/MulMul2residual_block_0/activation_381/Relu:activations:0=residual_block_0/spatial_dropout1d_191/dropout/Const:output:0*
T0*-
_output_shapes
:?????????24
2residual_block_0/spatial_dropout1d_191/dropout/MulΠ
Eresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2G
Eresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/shape/1
Cresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/shapePack=residual_block_0/spatial_dropout1d_191/strided_slice:output:0Nresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/shape/1:output:0?residual_block_0/spatial_dropout1d_191/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2E
Cresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/shapeΕ
Kresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/RandomUniformRandomUniformLresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02M
Kresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/RandomUniformΓ
=residual_block_0/spatial_dropout1d_191/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2?
=residual_block_0/spatial_dropout1d_191/dropout/GreaterEqual/yη
;residual_block_0/spatial_dropout1d_191/dropout/GreaterEqualGreaterEqualTresidual_block_0/spatial_dropout1d_191/dropout/random_uniform/RandomUniform:output:0Fresidual_block_0/spatial_dropout1d_191/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2=
;residual_block_0/spatial_dropout1d_191/dropout/GreaterEqual
3residual_block_0/spatial_dropout1d_191/dropout/CastCast?residual_block_0/spatial_dropout1d_191/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????25
3residual_block_0/spatial_dropout1d_191/dropout/Cast
4residual_block_0/spatial_dropout1d_191/dropout/Mul_1Mul6residual_block_0/spatial_dropout1d_191/dropout/Mul:z:07residual_block_0/spatial_dropout1d_191/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????26
4residual_block_0/spatial_dropout1d_191/dropout/Mul_1Ζ
$residual_block_0/activation_382/ReluRelu8residual_block_0/spatial_dropout1d_191/dropout/Mul_1:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_0/activation_382/Relu»
6residual_block_0/matching_conv1D/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????28
6residual_block_0/matching_conv1D/conv1d/ExpandDims/dimϊ
2residual_block_0/matching_conv1D/conv1d/ExpandDims
ExpandDimsinputs?residual_block_0/matching_conv1D/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????24
2residual_block_0/matching_conv1D/conv1d/ExpandDims
Cresidual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpLresidual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:*
dtype02E
Cresidual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOpΆ
8residual_block_0/matching_conv1D/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2:
8residual_block_0/matching_conv1D/conv1d/ExpandDims_1/dimΌ
4residual_block_0/matching_conv1D/conv1d/ExpandDims_1
ExpandDimsKresidual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOp:value:0Aresidual_block_0/matching_conv1D/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:26
4residual_block_0/matching_conv1D/conv1d/ExpandDims_1Ό
'residual_block_0/matching_conv1D/conv1dConv2D;residual_block_0/matching_conv1D/conv1d/ExpandDims:output:0=residual_block_0/matching_conv1D/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingSAME*
strides
2)
'residual_block_0/matching_conv1D/conv1dχ
/residual_block_0/matching_conv1D/conv1d/SqueezeSqueeze0residual_block_0/matching_conv1D/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????21
/residual_block_0/matching_conv1D/conv1d/Squeezeπ
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOp@residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp
(residual_block_0/matching_conv1D/BiasAddBiasAdd8residual_block_0/matching_conv1D/conv1d/Squeeze:output:0?residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2*
(residual_block_0/matching_conv1D/BiasAddά
residual_block_0/add/addAddV21residual_block_0/matching_conv1D/BiasAdd:output:02residual_block_0/activation_382/Relu:activations:0*
T0*-
_output_shapes
:?????????2
residual_block_0/add/addͺ
$residual_block_0/activation_383/ReluReluresidual_block_0/add/add:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_0/activation_383/ReluΉ
&residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ~               2(
&residual_block_1/conv1D_0/Pad/paddingsβ
residual_block_1/conv1D_0/PadPad2residual_block_0/activation_383/Relu:activations:0/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ώ2
residual_block_1/conv1D_0/Padͺ
.residual_block_1/conv1D_0/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:20
.residual_block_1/conv1D_0/conv1d/dilation_rateι
Mresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ώ2O
Mresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeϋ
Oresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2Q
Oresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsρ
Jresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsλ
Gresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2I
Gresidual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/cropsΔ
;residual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shapeΝ
8residual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8residual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddingsέ
/residual_block_1/conv1D_0/conv1d/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_0/Pad:output:0Dresidual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ21
/residual_block_1/conv1D_0/conv1d/SpaceToBatchND­
/residual_block_1/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_1/conv1D_0/conv1d/ExpandDims/dim
+residual_block_1/conv1D_0/conv1d/ExpandDims
ExpandDims8residual_block_1/conv1D_0/conv1d/SpaceToBatchND:output:08residual_block_1/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_1/conv1D_0/conv1d/ExpandDims
<residual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_1/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_1/conv1D_0/conv1d/ExpandDims_1/dim‘
-residual_block_1/conv1D_0/conv1d/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_1/conv1D_0/conv1d/ExpandDims_1‘
 residual_block_1/conv1D_0/conv1dConv2D4residual_block_1/conv1D_0/conv1d/ExpandDims:output:06residual_block_1/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_1/conv1D_0/conv1dβ
(residual_block_1/conv1D_0/conv1d/SqueezeSqueeze)residual_block_1/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_1/conv1D_0/conv1d/SqueezeΔ
;residual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shapeΗ
5residual_block_1/conv1D_0/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        27
5residual_block_1/conv1D_0/conv1d/BatchToSpaceND/cropsε
/residual_block_1/conv1D_0/conv1d/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_0/conv1d/Squeeze:output:0Dresidual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_0/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????21
/residual_block_1/conv1D_0/conv1d/BatchToSpaceNDΫ
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_1/conv1D_0/BiasAdd/ReadVariableOpύ
!residual_block_1/conv1D_0/BiasAddBiasAdd8residual_block_1/conv1D_0/conv1d/BatchToSpaceND:output:08residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_1/conv1D_0/BiasAddΈ
$residual_block_1/activation_384/ReluRelu*residual_block_1/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_1/activation_384/ReluΎ
,residual_block_1/spatial_dropout1d_192/ShapeShape2residual_block_1/activation_384/Relu:activations:0*
T0*
_output_shapes
:2.
,residual_block_1/spatial_dropout1d_192/ShapeΒ
:residual_block_1/spatial_dropout1d_192/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:residual_block_1/spatial_dropout1d_192/strided_slice/stackΖ
<residual_block_1/spatial_dropout1d_192/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_1/spatial_dropout1d_192/strided_slice/stack_1Ζ
<residual_block_1/spatial_dropout1d_192/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_1/spatial_dropout1d_192/strided_slice/stack_2Μ
4residual_block_1/spatial_dropout1d_192/strided_sliceStridedSlice5residual_block_1/spatial_dropout1d_192/Shape:output:0Cresidual_block_1/spatial_dropout1d_192/strided_slice/stack:output:0Eresidual_block_1/spatial_dropout1d_192/strided_slice/stack_1:output:0Eresidual_block_1/spatial_dropout1d_192/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4residual_block_1/spatial_dropout1d_192/strided_sliceΖ
<residual_block_1/spatial_dropout1d_192/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_1/spatial_dropout1d_192/strided_slice_1/stackΚ
>residual_block_1/spatial_dropout1d_192/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_1/spatial_dropout1d_192/strided_slice_1/stack_1Κ
>residual_block_1/spatial_dropout1d_192/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_1/spatial_dropout1d_192/strided_slice_1/stack_2Φ
6residual_block_1/spatial_dropout1d_192/strided_slice_1StridedSlice5residual_block_1/spatial_dropout1d_192/Shape:output:0Eresidual_block_1/spatial_dropout1d_192/strided_slice_1/stack:output:0Gresidual_block_1/spatial_dropout1d_192/strided_slice_1/stack_1:output:0Gresidual_block_1/spatial_dropout1d_192/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6residual_block_1/spatial_dropout1d_192/strided_slice_1±
4residual_block_1/spatial_dropout1d_192/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?26
4residual_block_1/spatial_dropout1d_192/dropout/Const
2residual_block_1/spatial_dropout1d_192/dropout/MulMul2residual_block_1/activation_384/Relu:activations:0=residual_block_1/spatial_dropout1d_192/dropout/Const:output:0*
T0*-
_output_shapes
:?????????24
2residual_block_1/spatial_dropout1d_192/dropout/MulΠ
Eresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2G
Eresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/shape/1
Cresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/shapePack=residual_block_1/spatial_dropout1d_192/strided_slice:output:0Nresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/shape/1:output:0?residual_block_1/spatial_dropout1d_192/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2E
Cresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/shapeΕ
Kresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/RandomUniformRandomUniformLresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02M
Kresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/RandomUniformΓ
=residual_block_1/spatial_dropout1d_192/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2?
=residual_block_1/spatial_dropout1d_192/dropout/GreaterEqual/yη
;residual_block_1/spatial_dropout1d_192/dropout/GreaterEqualGreaterEqualTresidual_block_1/spatial_dropout1d_192/dropout/random_uniform/RandomUniform:output:0Fresidual_block_1/spatial_dropout1d_192/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2=
;residual_block_1/spatial_dropout1d_192/dropout/GreaterEqual
3residual_block_1/spatial_dropout1d_192/dropout/CastCast?residual_block_1/spatial_dropout1d_192/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????25
3residual_block_1/spatial_dropout1d_192/dropout/Cast
4residual_block_1/spatial_dropout1d_192/dropout/Mul_1Mul6residual_block_1/spatial_dropout1d_192/dropout/Mul:z:07residual_block_1/spatial_dropout1d_192/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????26
4residual_block_1/spatial_dropout1d_192/dropout/Mul_1Ή
&residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ~               2(
&residual_block_1/conv1D_1/Pad/paddingsθ
residual_block_1/conv1D_1/PadPad8residual_block_1/spatial_dropout1d_192/dropout/Mul_1:z:0/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ώ2
residual_block_1/conv1D_1/Padͺ
.residual_block_1/conv1D_1/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:20
.residual_block_1/conv1D_1/conv1d/dilation_rateι
Mresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ώ2O
Mresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeϋ
Oresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2Q
Oresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsρ
Jresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsλ
Gresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2I
Gresidual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/cropsΔ
;residual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shapeΝ
8residual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8residual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddingsέ
/residual_block_1/conv1D_1/conv1d/SpaceToBatchNDSpaceToBatchND&residual_block_1/conv1D_1/Pad:output:0Dresidual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shape:output:0Aresidual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ21
/residual_block_1/conv1D_1/conv1d/SpaceToBatchND­
/residual_block_1/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_1/conv1D_1/conv1d/ExpandDims/dim
+residual_block_1/conv1D_1/conv1d/ExpandDims
ExpandDims8residual_block_1/conv1D_1/conv1d/SpaceToBatchND:output:08residual_block_1/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_1/conv1D_1/conv1d/ExpandDims
<residual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_1/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_1/conv1D_1/conv1d/ExpandDims_1/dim‘
-residual_block_1/conv1D_1/conv1d/ExpandDims_1
ExpandDimsDresidual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_1/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_1/conv1D_1/conv1d/ExpandDims_1‘
 residual_block_1/conv1D_1/conv1dConv2D4residual_block_1/conv1D_1/conv1d/ExpandDims:output:06residual_block_1/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_1/conv1D_1/conv1dβ
(residual_block_1/conv1D_1/conv1d/SqueezeSqueeze)residual_block_1/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_1/conv1D_1/conv1d/SqueezeΔ
;residual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shapeΗ
5residual_block_1/conv1D_1/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        27
5residual_block_1/conv1D_1/conv1d/BatchToSpaceND/cropsε
/residual_block_1/conv1D_1/conv1d/BatchToSpaceNDBatchToSpaceND1residual_block_1/conv1D_1/conv1d/Squeeze:output:0Dresidual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shape:output:0>residual_block_1/conv1D_1/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????21
/residual_block_1/conv1D_1/conv1d/BatchToSpaceNDΫ
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_1/conv1D_1/BiasAdd/ReadVariableOpύ
!residual_block_1/conv1D_1/BiasAddBiasAdd8residual_block_1/conv1D_1/conv1d/BatchToSpaceND:output:08residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_1/conv1D_1/BiasAddΈ
$residual_block_1/activation_385/ReluRelu*residual_block_1/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_1/activation_385/ReluΎ
,residual_block_1/spatial_dropout1d_193/ShapeShape2residual_block_1/activation_385/Relu:activations:0*
T0*
_output_shapes
:2.
,residual_block_1/spatial_dropout1d_193/ShapeΒ
:residual_block_1/spatial_dropout1d_193/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:residual_block_1/spatial_dropout1d_193/strided_slice/stackΖ
<residual_block_1/spatial_dropout1d_193/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_1/spatial_dropout1d_193/strided_slice/stack_1Ζ
<residual_block_1/spatial_dropout1d_193/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_1/spatial_dropout1d_193/strided_slice/stack_2Μ
4residual_block_1/spatial_dropout1d_193/strided_sliceStridedSlice5residual_block_1/spatial_dropout1d_193/Shape:output:0Cresidual_block_1/spatial_dropout1d_193/strided_slice/stack:output:0Eresidual_block_1/spatial_dropout1d_193/strided_slice/stack_1:output:0Eresidual_block_1/spatial_dropout1d_193/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4residual_block_1/spatial_dropout1d_193/strided_sliceΖ
<residual_block_1/spatial_dropout1d_193/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_1/spatial_dropout1d_193/strided_slice_1/stackΚ
>residual_block_1/spatial_dropout1d_193/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_1/spatial_dropout1d_193/strided_slice_1/stack_1Κ
>residual_block_1/spatial_dropout1d_193/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_1/spatial_dropout1d_193/strided_slice_1/stack_2Φ
6residual_block_1/spatial_dropout1d_193/strided_slice_1StridedSlice5residual_block_1/spatial_dropout1d_193/Shape:output:0Eresidual_block_1/spatial_dropout1d_193/strided_slice_1/stack:output:0Gresidual_block_1/spatial_dropout1d_193/strided_slice_1/stack_1:output:0Gresidual_block_1/spatial_dropout1d_193/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6residual_block_1/spatial_dropout1d_193/strided_slice_1±
4residual_block_1/spatial_dropout1d_193/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?26
4residual_block_1/spatial_dropout1d_193/dropout/Const
2residual_block_1/spatial_dropout1d_193/dropout/MulMul2residual_block_1/activation_385/Relu:activations:0=residual_block_1/spatial_dropout1d_193/dropout/Const:output:0*
T0*-
_output_shapes
:?????????24
2residual_block_1/spatial_dropout1d_193/dropout/MulΠ
Eresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2G
Eresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/shape/1
Cresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/shapePack=residual_block_1/spatial_dropout1d_193/strided_slice:output:0Nresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/shape/1:output:0?residual_block_1/spatial_dropout1d_193/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2E
Cresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/shapeΕ
Kresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/RandomUniformRandomUniformLresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02M
Kresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/RandomUniformΓ
=residual_block_1/spatial_dropout1d_193/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2?
=residual_block_1/spatial_dropout1d_193/dropout/GreaterEqual/yη
;residual_block_1/spatial_dropout1d_193/dropout/GreaterEqualGreaterEqualTresidual_block_1/spatial_dropout1d_193/dropout/random_uniform/RandomUniform:output:0Fresidual_block_1/spatial_dropout1d_193/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2=
;residual_block_1/spatial_dropout1d_193/dropout/GreaterEqual
3residual_block_1/spatial_dropout1d_193/dropout/CastCast?residual_block_1/spatial_dropout1d_193/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????25
3residual_block_1/spatial_dropout1d_193/dropout/Cast
4residual_block_1/spatial_dropout1d_193/dropout/Mul_1Mul6residual_block_1/spatial_dropout1d_193/dropout/Mul:z:07residual_block_1/spatial_dropout1d_193/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????26
4residual_block_1/spatial_dropout1d_193/dropout/Mul_1Ζ
$residual_block_1/activation_386/ReluRelu8residual_block_1/spatial_dropout1d_193/dropout/Mul_1:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_1/activation_386/Reluα
residual_block_1/add_1/addAddV22residual_block_0/activation_383/Relu:activations:02residual_block_1/activation_386/Relu:activations:0*
T0*-
_output_shapes
:?????????2
residual_block_1/add_1/add¬
$residual_block_1/activation_387/ReluReluresidual_block_1/add_1/add:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_1/activation_387/ReluΉ
&residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ό               2(
&residual_block_2/conv1D_0/Pad/paddingsβ
residual_block_2/conv1D_0/PadPad2residual_block_1/activation_387/Relu:activations:0/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ό2
residual_block_2/conv1D_0/Padͺ
.residual_block_2/conv1D_0/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:20
.residual_block_2/conv1D_0/conv1d/dilation_rateι
Mresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ό2O
Mresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeϋ
Oresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2Q
Oresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsρ
Jresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsλ
Gresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2I
Gresidual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/cropsΔ
;residual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shapeΝ
8residual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8residual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddingsέ
/residual_block_2/conv1D_0/conv1d/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_0/Pad:output:0Dresidual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ21
/residual_block_2/conv1D_0/conv1d/SpaceToBatchND­
/residual_block_2/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_2/conv1D_0/conv1d/ExpandDims/dim
+residual_block_2/conv1D_0/conv1d/ExpandDims
ExpandDims8residual_block_2/conv1D_0/conv1d/SpaceToBatchND:output:08residual_block_2/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_2/conv1D_0/conv1d/ExpandDims
<residual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_2/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_2/conv1D_0/conv1d/ExpandDims_1/dim‘
-residual_block_2/conv1D_0/conv1d/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_2/conv1D_0/conv1d/ExpandDims_1‘
 residual_block_2/conv1D_0/conv1dConv2D4residual_block_2/conv1D_0/conv1d/ExpandDims:output:06residual_block_2/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_2/conv1D_0/conv1dβ
(residual_block_2/conv1D_0/conv1d/SqueezeSqueeze)residual_block_2/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_2/conv1D_0/conv1d/SqueezeΔ
;residual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shapeΗ
5residual_block_2/conv1D_0/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        27
5residual_block_2/conv1D_0/conv1d/BatchToSpaceND/cropsε
/residual_block_2/conv1D_0/conv1d/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_0/conv1d/Squeeze:output:0Dresidual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_0/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????21
/residual_block_2/conv1D_0/conv1d/BatchToSpaceNDΫ
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_2/conv1D_0/BiasAdd/ReadVariableOpύ
!residual_block_2/conv1D_0/BiasAddBiasAdd8residual_block_2/conv1D_0/conv1d/BatchToSpaceND:output:08residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_2/conv1D_0/BiasAddΈ
$residual_block_2/activation_388/ReluRelu*residual_block_2/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_2/activation_388/ReluΎ
,residual_block_2/spatial_dropout1d_194/ShapeShape2residual_block_2/activation_388/Relu:activations:0*
T0*
_output_shapes
:2.
,residual_block_2/spatial_dropout1d_194/ShapeΒ
:residual_block_2/spatial_dropout1d_194/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:residual_block_2/spatial_dropout1d_194/strided_slice/stackΖ
<residual_block_2/spatial_dropout1d_194/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_2/spatial_dropout1d_194/strided_slice/stack_1Ζ
<residual_block_2/spatial_dropout1d_194/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_2/spatial_dropout1d_194/strided_slice/stack_2Μ
4residual_block_2/spatial_dropout1d_194/strided_sliceStridedSlice5residual_block_2/spatial_dropout1d_194/Shape:output:0Cresidual_block_2/spatial_dropout1d_194/strided_slice/stack:output:0Eresidual_block_2/spatial_dropout1d_194/strided_slice/stack_1:output:0Eresidual_block_2/spatial_dropout1d_194/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4residual_block_2/spatial_dropout1d_194/strided_sliceΖ
<residual_block_2/spatial_dropout1d_194/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_2/spatial_dropout1d_194/strided_slice_1/stackΚ
>residual_block_2/spatial_dropout1d_194/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_2/spatial_dropout1d_194/strided_slice_1/stack_1Κ
>residual_block_2/spatial_dropout1d_194/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_2/spatial_dropout1d_194/strided_slice_1/stack_2Φ
6residual_block_2/spatial_dropout1d_194/strided_slice_1StridedSlice5residual_block_2/spatial_dropout1d_194/Shape:output:0Eresidual_block_2/spatial_dropout1d_194/strided_slice_1/stack:output:0Gresidual_block_2/spatial_dropout1d_194/strided_slice_1/stack_1:output:0Gresidual_block_2/spatial_dropout1d_194/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6residual_block_2/spatial_dropout1d_194/strided_slice_1±
4residual_block_2/spatial_dropout1d_194/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?26
4residual_block_2/spatial_dropout1d_194/dropout/Const
2residual_block_2/spatial_dropout1d_194/dropout/MulMul2residual_block_2/activation_388/Relu:activations:0=residual_block_2/spatial_dropout1d_194/dropout/Const:output:0*
T0*-
_output_shapes
:?????????24
2residual_block_2/spatial_dropout1d_194/dropout/MulΠ
Eresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2G
Eresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/shape/1
Cresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/shapePack=residual_block_2/spatial_dropout1d_194/strided_slice:output:0Nresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/shape/1:output:0?residual_block_2/spatial_dropout1d_194/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2E
Cresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/shapeΕ
Kresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/RandomUniformRandomUniformLresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02M
Kresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/RandomUniformΓ
=residual_block_2/spatial_dropout1d_194/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2?
=residual_block_2/spatial_dropout1d_194/dropout/GreaterEqual/yη
;residual_block_2/spatial_dropout1d_194/dropout/GreaterEqualGreaterEqualTresidual_block_2/spatial_dropout1d_194/dropout/random_uniform/RandomUniform:output:0Fresidual_block_2/spatial_dropout1d_194/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2=
;residual_block_2/spatial_dropout1d_194/dropout/GreaterEqual
3residual_block_2/spatial_dropout1d_194/dropout/CastCast?residual_block_2/spatial_dropout1d_194/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????25
3residual_block_2/spatial_dropout1d_194/dropout/Cast
4residual_block_2/spatial_dropout1d_194/dropout/Mul_1Mul6residual_block_2/spatial_dropout1d_194/dropout/Mul:z:07residual_block_2/spatial_dropout1d_194/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????26
4residual_block_2/spatial_dropout1d_194/dropout/Mul_1Ή
&residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ό               2(
&residual_block_2/conv1D_1/Pad/paddingsθ
residual_block_2/conv1D_1/PadPad8residual_block_2/spatial_dropout1d_194/dropout/Mul_1:z:0/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ό2
residual_block_2/conv1D_1/Padͺ
.residual_block_2/conv1D_1/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:20
.residual_block_2/conv1D_1/conv1d/dilation_rateι
Mresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ό2O
Mresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeϋ
Oresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2Q
Oresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsρ
Jresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsλ
Gresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2I
Gresidual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/cropsΔ
;residual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shapeΝ
8residual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2:
8residual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddingsέ
/residual_block_2/conv1D_1/conv1d/SpaceToBatchNDSpaceToBatchND&residual_block_2/conv1D_1/Pad:output:0Dresidual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shape:output:0Aresidual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ21
/residual_block_2/conv1D_1/conv1d/SpaceToBatchND­
/residual_block_2/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????21
/residual_block_2/conv1D_1/conv1d/ExpandDims/dim
+residual_block_2/conv1D_1/conv1d/ExpandDims
ExpandDims8residual_block_2/conv1D_1/conv1d/SpaceToBatchND:output:08residual_block_2/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2-
+residual_block_2/conv1D_1/conv1d/ExpandDims
<residual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEresidual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02>
<residual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp¨
1residual_block_2/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1residual_block_2/conv1D_1/conv1d/ExpandDims_1/dim‘
-residual_block_2/conv1D_1/conv1d/ExpandDims_1
ExpandDimsDresidual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0:residual_block_2/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2/
-residual_block_2/conv1D_1/conv1d/ExpandDims_1‘
 residual_block_2/conv1D_1/conv1dConv2D4residual_block_2/conv1D_1/conv1d/ExpandDims:output:06residual_block_2/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
2"
 residual_block_2/conv1D_1/conv1dβ
(residual_block_2/conv1D_1/conv1d/SqueezeSqueeze)residual_block_2/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2*
(residual_block_2/conv1D_1/conv1d/SqueezeΔ
;residual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2=
;residual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shapeΗ
5residual_block_2/conv1D_1/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        27
5residual_block_2/conv1D_1/conv1d/BatchToSpaceND/cropsε
/residual_block_2/conv1D_1/conv1d/BatchToSpaceNDBatchToSpaceND1residual_block_2/conv1D_1/conv1d/Squeeze:output:0Dresidual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shape:output:0>residual_block_2/conv1D_1/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????21
/residual_block_2/conv1D_1/conv1d/BatchToSpaceNDΫ
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOp9residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype022
0residual_block_2/conv1D_1/BiasAdd/ReadVariableOpύ
!residual_block_2/conv1D_1/BiasAddBiasAdd8residual_block_2/conv1D_1/conv1d/BatchToSpaceND:output:08residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2#
!residual_block_2/conv1D_1/BiasAddΈ
$residual_block_2/activation_389/ReluRelu*residual_block_2/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2&
$residual_block_2/activation_389/ReluΎ
,residual_block_2/spatial_dropout1d_195/ShapeShape2residual_block_2/activation_389/Relu:activations:0*
T0*
_output_shapes
:2.
,residual_block_2/spatial_dropout1d_195/ShapeΒ
:residual_block_2/spatial_dropout1d_195/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:residual_block_2/spatial_dropout1d_195/strided_slice/stackΖ
<residual_block_2/spatial_dropout1d_195/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_2/spatial_dropout1d_195/strided_slice/stack_1Ζ
<residual_block_2/spatial_dropout1d_195/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_2/spatial_dropout1d_195/strided_slice/stack_2Μ
4residual_block_2/spatial_dropout1d_195/strided_sliceStridedSlice5residual_block_2/spatial_dropout1d_195/Shape:output:0Cresidual_block_2/spatial_dropout1d_195/strided_slice/stack:output:0Eresidual_block_2/spatial_dropout1d_195/strided_slice/stack_1:output:0Eresidual_block_2/spatial_dropout1d_195/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4residual_block_2/spatial_dropout1d_195/strided_sliceΖ
<residual_block_2/spatial_dropout1d_195/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<residual_block_2/spatial_dropout1d_195/strided_slice_1/stackΚ
>residual_block_2/spatial_dropout1d_195/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_2/spatial_dropout1d_195/strided_slice_1/stack_1Κ
>residual_block_2/spatial_dropout1d_195/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>residual_block_2/spatial_dropout1d_195/strided_slice_1/stack_2Φ
6residual_block_2/spatial_dropout1d_195/strided_slice_1StridedSlice5residual_block_2/spatial_dropout1d_195/Shape:output:0Eresidual_block_2/spatial_dropout1d_195/strided_slice_1/stack:output:0Gresidual_block_2/spatial_dropout1d_195/strided_slice_1/stack_1:output:0Gresidual_block_2/spatial_dropout1d_195/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6residual_block_2/spatial_dropout1d_195/strided_slice_1±
4residual_block_2/spatial_dropout1d_195/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *’Ό?26
4residual_block_2/spatial_dropout1d_195/dropout/Const
2residual_block_2/spatial_dropout1d_195/dropout/MulMul2residual_block_2/activation_389/Relu:activations:0=residual_block_2/spatial_dropout1d_195/dropout/Const:output:0*
T0*-
_output_shapes
:?????????24
2residual_block_2/spatial_dropout1d_195/dropout/MulΠ
Eresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2G
Eresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/shape/1
Cresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/shapePack=residual_block_2/spatial_dropout1d_195/strided_slice:output:0Nresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/shape/1:output:0?residual_block_2/spatial_dropout1d_195/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2E
Cresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/shapeΕ
Kresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/RandomUniformRandomUniformLresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :??????????????????*
dtype02M
Kresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/RandomUniformΓ
=residual_block_2/spatial_dropout1d_195/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL=2?
=residual_block_2/spatial_dropout1d_195/dropout/GreaterEqual/yη
;residual_block_2/spatial_dropout1d_195/dropout/GreaterEqualGreaterEqualTresidual_block_2/spatial_dropout1d_195/dropout/random_uniform/RandomUniform:output:0Fresidual_block_2/spatial_dropout1d_195/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????2=
;residual_block_2/spatial_dropout1d_195/dropout/GreaterEqual
3residual_block_2/spatial_dropout1d_195/dropout/CastCast?residual_block_2/spatial_dropout1d_195/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????25
3residual_block_2/spatial_dropout1d_195/dropout/Cast
4residual_block_2/spatial_dropout1d_195/dropout/Mul_1Mul6residual_block_2/spatial_dropout1d_195/dropout/Mul:z:07residual_block_2/spatial_dropout1d_195/dropout/Cast:y:0*
T0*-
_output_shapes
:?????????26
4residual_block_2/spatial_dropout1d_195/dropout/Mul_1Ζ
$residual_block_2/activation_390/ReluRelu8residual_block_2/spatial_dropout1d_195/dropout/Mul_1:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_2/activation_390/Reluα
residual_block_2/add_2/addAddV22residual_block_1/activation_387/Relu:activations:02residual_block_2/activation_390/Relu:activations:0*
T0*-
_output_shapes
:?????????2
residual_block_2/add_2/add¬
$residual_block_2/activation_391/ReluReluresidual_block_2/add_2/add:z:0*
T0*-
_output_shapes
:?????????2&
$residual_block_2/activation_391/ReluΏ
	add_3/addAddV22residual_block_0/activation_382/Relu:activations:02residual_block_1/activation_386/Relu:activations:0*
T0*-
_output_shapes
:?????????2
	add_3/add
add_3/add_1AddV2add_3/add:z:02residual_block_2/activation_390/Relu:activations:0*
T0*-
_output_shapes
:?????????2
add_3/add_1
lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    2
lambda_17/strided_slice/stack
lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2!
lambda_17/strided_slice/stack_1
lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2!
lambda_17/strided_slice/stack_2Ι
lambda_17/strided_sliceStridedSliceadd_3/add_1:z:0&lambda_17/strided_slice/stack:output:0(lambda_17/strided_slice/stack_1:output:0(lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
lambda_17/strided_sliceu
IdentityIdentity lambda_17/strided_slice:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:?????????:::::::::::::::T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
λξ
―
"__inference__wrapped_model_1154918
tcn_17_input^
Zsequential_17_tcn_17_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resourceR
Nsequential_17_tcn_17_residual_block_0_conv1d_0_biasadd_readvariableop_resource^
Zsequential_17_tcn_17_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resourceR
Nsequential_17_tcn_17_residual_block_0_conv1d_1_biasadd_readvariableop_resourcee
asequential_17_tcn_17_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resourceY
Usequential_17_tcn_17_residual_block_0_matching_conv1d_biasadd_readvariableop_resource^
Zsequential_17_tcn_17_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resourceR
Nsequential_17_tcn_17_residual_block_1_conv1d_0_biasadd_readvariableop_resource^
Zsequential_17_tcn_17_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resourceR
Nsequential_17_tcn_17_residual_block_1_conv1d_1_biasadd_readvariableop_resource^
Zsequential_17_tcn_17_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resourceR
Nsequential_17_tcn_17_residual_block_2_conv1d_0_biasadd_readvariableop_resource^
Zsequential_17_tcn_17_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resourceR
Nsequential_17_tcn_17_residual_block_2_conv1d_1_biasadd_readvariableop_resource9
5sequential_17_dense_34_matmul_readvariableop_resource:
6sequential_17_dense_34_biasadd_readvariableop_resource9
5sequential_17_dense_35_matmul_readvariableop_resource:
6sequential_17_dense_35_biasadd_readvariableop_resource
identityγ
;sequential_17/tcn_17/residual_block_0/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ?               2=
;sequential_17/tcn_17/residual_block_0/conv1D_0/Pad/paddingsϊ
2sequential_17/tcn_17/residual_block_0/conv1D_0/PadPadtcn_17_inputDsequential_17/tcn_17/residual_block_0/conv1D_0/Pad/paddings:output:0*
T0*,
_output_shapes
:?????????Ώ24
2sequential_17/tcn_17/residual_block_0/conv1D_0/PadΧ
Dsequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2F
Dsequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims/dimΩ
@sequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims
ExpandDims;sequential_17/tcn_17/residual_block_0/conv1D_0/Pad:output:0Msequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????Ώ2B
@sequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDimsΖ
Qsequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpZsequential_17_tcn_17_residual_block_0_conv1d_0_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@*
dtype02S
Qsequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp?
Fsequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fsequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/dimτ
Bsequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1
ExpandDimsYsequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0Osequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@2D
Bsequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1υ
5sequential_17/tcn_17/residual_block_0/conv1D_0/conv1dConv2DIsequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims:output:0Ksequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
27
5sequential_17/tcn_17/residual_block_0/conv1D_0/conv1d‘
=sequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/SqueezeSqueeze>sequential_17/tcn_17/residual_block_0/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2?
=sequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/Squeeze
Esequential_17/tcn_17/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpNsequential_17_tcn_17_residual_block_0_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02G
Esequential_17/tcn_17/residual_block_0/conv1D_0/BiasAdd/ReadVariableOpΚ
6sequential_17/tcn_17/residual_block_0/conv1D_0/BiasAddBiasAddFsequential_17/tcn_17/residual_block_0/conv1D_0/conv1d/Squeeze:output:0Msequential_17/tcn_17/residual_block_0/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????28
6sequential_17/tcn_17/residual_block_0/conv1D_0/BiasAddχ
9sequential_17/tcn_17/residual_block_0/activation_380/ReluRelu?sequential_17/tcn_17/residual_block_0/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2;
9sequential_17/tcn_17/residual_block_0/activation_380/Relu
Dsequential_17/tcn_17/residual_block_0/spatial_dropout1d_190/IdentityIdentityGsequential_17/tcn_17/residual_block_0/activation_380/Relu:activations:0*
T0*-
_output_shapes
:?????????2F
Dsequential_17/tcn_17/residual_block_0/spatial_dropout1d_190/Identityγ
;sequential_17/tcn_17/residual_block_0/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ?               2=
;sequential_17/tcn_17/residual_block_0/conv1D_1/Pad/paddingsΌ
2sequential_17/tcn_17/residual_block_0/conv1D_1/PadPadMsequential_17/tcn_17/residual_block_0/spatial_dropout1d_190/Identity:output:0Dsequential_17/tcn_17/residual_block_0/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ24
2sequential_17/tcn_17/residual_block_0/conv1D_1/PadΧ
Dsequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2F
Dsequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims/dimΪ
@sequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims
ExpandDims;sequential_17/tcn_17/residual_block_0/conv1D_1/Pad:output:0Msequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2B
@sequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDimsΗ
Qsequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpZsequential_17_tcn_17_residual_block_0_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02S
Qsequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?
Fsequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fsequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/dimυ
Bsequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1
ExpandDimsYsequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Osequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2D
Bsequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1υ
5sequential_17/tcn_17/residual_block_0/conv1D_1/conv1dConv2DIsequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims:output:0Ksequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
27
5sequential_17/tcn_17/residual_block_0/conv1D_1/conv1d‘
=sequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/SqueezeSqueeze>sequential_17/tcn_17/residual_block_0/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2?
=sequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/Squeeze
Esequential_17/tcn_17/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpNsequential_17_tcn_17_residual_block_0_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02G
Esequential_17/tcn_17/residual_block_0/conv1D_1/BiasAdd/ReadVariableOpΚ
6sequential_17/tcn_17/residual_block_0/conv1D_1/BiasAddBiasAddFsequential_17/tcn_17/residual_block_0/conv1D_1/conv1d/Squeeze:output:0Msequential_17/tcn_17/residual_block_0/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????28
6sequential_17/tcn_17/residual_block_0/conv1D_1/BiasAddχ
9sequential_17/tcn_17/residual_block_0/activation_381/ReluRelu?sequential_17/tcn_17/residual_block_0/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2;
9sequential_17/tcn_17/residual_block_0/activation_381/Relu
Dsequential_17/tcn_17/residual_block_0/spatial_dropout1d_191/IdentityIdentityGsequential_17/tcn_17/residual_block_0/activation_381/Relu:activations:0*
T0*-
_output_shapes
:?????????2F
Dsequential_17/tcn_17/residual_block_0/spatial_dropout1d_191/Identity
9sequential_17/tcn_17/residual_block_0/activation_382/ReluReluMsequential_17/tcn_17/residual_block_0/spatial_dropout1d_191/Identity:output:0*
T0*-
_output_shapes
:?????????2;
9sequential_17/tcn_17/residual_block_0/activation_382/Reluε
Ksequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2M
Ksequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims/dimΏ
Gsequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims
ExpandDimstcn_17_inputTsequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????2I
Gsequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDimsΫ
Xsequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpasequential_17_tcn_17_residual_block_0_matching_conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:*
dtype02Z
Xsequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOpΰ
Msequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2O
Msequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/dim
Isequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1
ExpandDims`sequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/ReadVariableOp:value:0Vsequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:2K
Isequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1
<sequential_17/tcn_17/residual_block_0/matching_conv1D/conv1dConv2DPsequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims:output:0Rsequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingSAME*
strides
2>
<sequential_17/tcn_17/residual_block_0/matching_conv1D/conv1dΆ
Dsequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/SqueezeSqueezeEsequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2F
Dsequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/Squeeze―
Lsequential_17/tcn_17/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpReadVariableOpUsequential_17_tcn_17_residual_block_0_matching_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02N
Lsequential_17/tcn_17/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOpζ
=sequential_17/tcn_17/residual_block_0/matching_conv1D/BiasAddBiasAddMsequential_17/tcn_17/residual_block_0/matching_conv1D/conv1d/Squeeze:output:0Tsequential_17/tcn_17/residual_block_0/matching_conv1D/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????2?
=sequential_17/tcn_17/residual_block_0/matching_conv1D/BiasAdd°
-sequential_17/tcn_17/residual_block_0/add/addAddV2Fsequential_17/tcn_17/residual_block_0/matching_conv1D/BiasAdd:output:0Gsequential_17/tcn_17/residual_block_0/activation_382/Relu:activations:0*
T0*-
_output_shapes
:?????????2/
-sequential_17/tcn_17/residual_block_0/add/addι
9sequential_17/tcn_17/residual_block_0/activation_383/ReluRelu1sequential_17/tcn_17/residual_block_0/add/add:z:0*
T0*-
_output_shapes
:?????????2;
9sequential_17/tcn_17/residual_block_0/activation_383/Reluγ
;sequential_17/tcn_17/residual_block_1/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ~               2=
;sequential_17/tcn_17/residual_block_1/conv1D_0/Pad/paddingsΆ
2sequential_17/tcn_17/residual_block_1/conv1D_0/PadPadGsequential_17/tcn_17/residual_block_0/activation_383/Relu:activations:0Dsequential_17/tcn_17/residual_block_1/conv1D_0/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ώ24
2sequential_17/tcn_17/residual_block_1/conv1D_0/PadΤ
Csequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/dilation_rate
bsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ώ2d
bsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/input_shape₯
dsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2f
dsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddings
_sequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2a
_sequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/paddings
\sequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2^
\sequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/required_space_to_batch_paddings/cropsξ
Psequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2R
Psequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shapeχ
Msequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2O
Msequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddingsΖ
Dsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchNDSpaceToBatchND;sequential_17/tcn_17/residual_block_1/conv1D_0/Pad:output:0Ysequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/block_shape:output:0Vsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ2F
Dsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchNDΧ
Dsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2F
Dsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims/dimμ
@sequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims
ExpandDimsMsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/SpaceToBatchND:output:0Msequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2B
@sequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDimsΗ
Qsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpZsequential_17_tcn_17_residual_block_1_conv1d_0_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02S
Qsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp?
Fsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/dimυ
Bsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1
ExpandDimsYsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0Osequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2D
Bsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1υ
5sequential_17/tcn_17/residual_block_1/conv1D_0/conv1dConv2DIsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims:output:0Ksequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
27
5sequential_17/tcn_17/residual_block_1/conv1D_0/conv1d‘
=sequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/SqueezeSqueeze>sequential_17/tcn_17/residual_block_1/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2?
=sequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/Squeezeξ
Psequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2R
Psequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shapeρ
Jsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/cropsΞ
Dsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceNDBatchToSpaceNDFsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/Squeeze:output:0Ysequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/block_shape:output:0Ssequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????2F
Dsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND
Esequential_17/tcn_17/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpNsequential_17_tcn_17_residual_block_1_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02G
Esequential_17/tcn_17/residual_block_1/conv1D_0/BiasAdd/ReadVariableOpΡ
6sequential_17/tcn_17/residual_block_1/conv1D_0/BiasAddBiasAddMsequential_17/tcn_17/residual_block_1/conv1D_0/conv1d/BatchToSpaceND:output:0Msequential_17/tcn_17/residual_block_1/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????28
6sequential_17/tcn_17/residual_block_1/conv1D_0/BiasAddχ
9sequential_17/tcn_17/residual_block_1/activation_384/ReluRelu?sequential_17/tcn_17/residual_block_1/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2;
9sequential_17/tcn_17/residual_block_1/activation_384/Relu
Dsequential_17/tcn_17/residual_block_1/spatial_dropout1d_192/IdentityIdentityGsequential_17/tcn_17/residual_block_1/activation_384/Relu:activations:0*
T0*-
_output_shapes
:?????????2F
Dsequential_17/tcn_17/residual_block_1/spatial_dropout1d_192/Identityγ
;sequential_17/tcn_17/residual_block_1/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ~               2=
;sequential_17/tcn_17/residual_block_1/conv1D_1/Pad/paddingsΌ
2sequential_17/tcn_17/residual_block_1/conv1D_1/PadPadMsequential_17/tcn_17/residual_block_1/spatial_dropout1d_192/Identity:output:0Dsequential_17/tcn_17/residual_block_1/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ώ24
2sequential_17/tcn_17/residual_block_1/conv1D_1/PadΤ
Csequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/dilation_rate
bsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ώ2d
bsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/input_shape₯
dsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2f
dsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddings
_sequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2a
_sequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/paddings
\sequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2^
\sequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/required_space_to_batch_paddings/cropsξ
Psequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2R
Psequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shapeχ
Msequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2O
Msequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddingsΖ
Dsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchNDSpaceToBatchND;sequential_17/tcn_17/residual_block_1/conv1D_1/Pad:output:0Ysequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/block_shape:output:0Vsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ2F
Dsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchNDΧ
Dsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2F
Dsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims/dimμ
@sequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims
ExpandDimsMsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/SpaceToBatchND:output:0Msequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2B
@sequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDimsΗ
Qsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpZsequential_17_tcn_17_residual_block_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02S
Qsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?
Fsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/dimυ
Bsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1
ExpandDimsYsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Osequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2D
Bsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1υ
5sequential_17/tcn_17/residual_block_1/conv1D_1/conv1dConv2DIsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims:output:0Ksequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
27
5sequential_17/tcn_17/residual_block_1/conv1D_1/conv1d‘
=sequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/SqueezeSqueeze>sequential_17/tcn_17/residual_block_1/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2?
=sequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/Squeezeξ
Psequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2R
Psequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shapeρ
Jsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/cropsΞ
Dsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceNDBatchToSpaceNDFsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/Squeeze:output:0Ysequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/block_shape:output:0Ssequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????2F
Dsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND
Esequential_17/tcn_17/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpNsequential_17_tcn_17_residual_block_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02G
Esequential_17/tcn_17/residual_block_1/conv1D_1/BiasAdd/ReadVariableOpΡ
6sequential_17/tcn_17/residual_block_1/conv1D_1/BiasAddBiasAddMsequential_17/tcn_17/residual_block_1/conv1D_1/conv1d/BatchToSpaceND:output:0Msequential_17/tcn_17/residual_block_1/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????28
6sequential_17/tcn_17/residual_block_1/conv1D_1/BiasAddχ
9sequential_17/tcn_17/residual_block_1/activation_385/ReluRelu?sequential_17/tcn_17/residual_block_1/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2;
9sequential_17/tcn_17/residual_block_1/activation_385/Relu
Dsequential_17/tcn_17/residual_block_1/spatial_dropout1d_193/IdentityIdentityGsequential_17/tcn_17/residual_block_1/activation_385/Relu:activations:0*
T0*-
_output_shapes
:?????????2F
Dsequential_17/tcn_17/residual_block_1/spatial_dropout1d_193/Identity
9sequential_17/tcn_17/residual_block_1/activation_386/ReluReluMsequential_17/tcn_17/residual_block_1/spatial_dropout1d_193/Identity:output:0*
T0*-
_output_shapes
:?????????2;
9sequential_17/tcn_17/residual_block_1/activation_386/Relu΅
/sequential_17/tcn_17/residual_block_1/add_1/addAddV2Gsequential_17/tcn_17/residual_block_0/activation_383/Relu:activations:0Gsequential_17/tcn_17/residual_block_1/activation_386/Relu:activations:0*
T0*-
_output_shapes
:?????????21
/sequential_17/tcn_17/residual_block_1/add_1/addλ
9sequential_17/tcn_17/residual_block_1/activation_387/ReluRelu3sequential_17/tcn_17/residual_block_1/add_1/add:z:0*
T0*-
_output_shapes
:?????????2;
9sequential_17/tcn_17/residual_block_1/activation_387/Reluγ
;sequential_17/tcn_17/residual_block_2/conv1D_0/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ό               2=
;sequential_17/tcn_17/residual_block_2/conv1D_0/Pad/paddingsΆ
2sequential_17/tcn_17/residual_block_2/conv1D_0/PadPadGsequential_17/tcn_17/residual_block_1/activation_387/Relu:activations:0Dsequential_17/tcn_17/residual_block_2/conv1D_0/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ό24
2sequential_17/tcn_17/residual_block_2/conv1D_0/PadΤ
Csequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/dilation_rate
bsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ό2d
bsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/input_shape₯
dsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2f
dsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/base_paddings
_sequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2a
_sequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/paddings
\sequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2^
\sequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/required_space_to_batch_paddings/cropsξ
Psequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2R
Psequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shapeχ
Msequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2O
Msequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddingsΖ
Dsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchNDSpaceToBatchND;sequential_17/tcn_17/residual_block_2/conv1D_0/Pad:output:0Ysequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/block_shape:output:0Vsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ2F
Dsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchNDΧ
Dsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2F
Dsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims/dimμ
@sequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims
ExpandDimsMsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/SpaceToBatchND:output:0Msequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2B
@sequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDimsΗ
Qsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpZsequential_17_tcn_17_residual_block_2_conv1d_0_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02S
Qsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp?
Fsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/dimυ
Bsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1
ExpandDimsYsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/ReadVariableOp:value:0Osequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2D
Bsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1υ
5sequential_17/tcn_17/residual_block_2/conv1D_0/conv1dConv2DIsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims:output:0Ksequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
27
5sequential_17/tcn_17/residual_block_2/conv1D_0/conv1d‘
=sequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/SqueezeSqueeze>sequential_17/tcn_17/residual_block_2/conv1D_0/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2?
=sequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/Squeezeξ
Psequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2R
Psequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shapeρ
Jsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/cropsΞ
Dsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceNDBatchToSpaceNDFsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/Squeeze:output:0Ysequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/block_shape:output:0Ssequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????2F
Dsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND
Esequential_17/tcn_17/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpReadVariableOpNsequential_17_tcn_17_residual_block_2_conv1d_0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02G
Esequential_17/tcn_17/residual_block_2/conv1D_0/BiasAdd/ReadVariableOpΡ
6sequential_17/tcn_17/residual_block_2/conv1D_0/BiasAddBiasAddMsequential_17/tcn_17/residual_block_2/conv1D_0/conv1d/BatchToSpaceND:output:0Msequential_17/tcn_17/residual_block_2/conv1D_0/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????28
6sequential_17/tcn_17/residual_block_2/conv1D_0/BiasAddχ
9sequential_17/tcn_17/residual_block_2/activation_388/ReluRelu?sequential_17/tcn_17/residual_block_2/conv1D_0/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2;
9sequential_17/tcn_17/residual_block_2/activation_388/Relu
Dsequential_17/tcn_17/residual_block_2/spatial_dropout1d_194/IdentityIdentityGsequential_17/tcn_17/residual_block_2/activation_388/Relu:activations:0*
T0*-
_output_shapes
:?????????2F
Dsequential_17/tcn_17/residual_block_2/spatial_dropout1d_194/Identityγ
;sequential_17/tcn_17/residual_block_2/conv1D_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*1
value(B&"        ό               2=
;sequential_17/tcn_17/residual_block_2/conv1D_1/Pad/paddingsΌ
2sequential_17/tcn_17/residual_block_2/conv1D_1/PadPadMsequential_17/tcn_17/residual_block_2/spatial_dropout1d_194/Identity:output:0Dsequential_17/tcn_17/residual_block_2/conv1D_1/Pad/paddings:output:0*
T0*-
_output_shapes
:?????????ό24
2sequential_17/tcn_17/residual_block_2/conv1D_1/PadΤ
Csequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/dilation_rate
bsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB:ό2d
bsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/input_shape₯
dsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2f
dsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/base_paddings
_sequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2a
_sequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/paddings
\sequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2^
\sequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/required_space_to_batch_paddings/cropsξ
Psequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2R
Psequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shapeχ
Msequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*!
valueB"        2O
Msequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddingsΖ
Dsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchNDSpaceToBatchND;sequential_17/tcn_17/residual_block_2/conv1D_1/Pad:output:0Ysequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/block_shape:output:0Vsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND/paddings:output:0*
T0*-
_output_shapes
:?????????Ώ2F
Dsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchNDΧ
Dsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????2F
Dsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims/dimμ
@sequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims
ExpandDimsMsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/SpaceToBatchND:output:0Msequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:?????????Ώ2B
@sequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDimsΗ
Qsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpZsequential_17_tcn_17_residual_block_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:@*
dtype02S
Qsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?
Fsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/dimυ
Bsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1
ExpandDimsYsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Osequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:@2D
Bsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1υ
5sequential_17/tcn_17/residual_block_2/conv1D_1/conv1dConv2DIsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims:output:0Ksequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:?????????*
paddingVALID*
strides
27
5sequential_17/tcn_17/residual_block_2/conv1D_1/conv1d‘
=sequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/SqueezeSqueeze>sequential_17/tcn_17/residual_block_2/conv1D_1/conv1d:output:0*
T0*-
_output_shapes
:?????????*
squeeze_dims

ύ????????2?
=sequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/Squeezeξ
Psequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB:2R
Psequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shapeρ
Jsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*!
valueB"        2L
Jsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/cropsΞ
Dsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceNDBatchToSpaceNDFsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/Squeeze:output:0Ysequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/block_shape:output:0Ssequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND/crops:output:0*
T0*-
_output_shapes
:?????????2F
Dsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND
Esequential_17/tcn_17/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpReadVariableOpNsequential_17_tcn_17_residual_block_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02G
Esequential_17/tcn_17/residual_block_2/conv1D_1/BiasAdd/ReadVariableOpΡ
6sequential_17/tcn_17/residual_block_2/conv1D_1/BiasAddBiasAddMsequential_17/tcn_17/residual_block_2/conv1D_1/conv1d/BatchToSpaceND:output:0Msequential_17/tcn_17/residual_block_2/conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:?????????28
6sequential_17/tcn_17/residual_block_2/conv1D_1/BiasAddχ
9sequential_17/tcn_17/residual_block_2/activation_389/ReluRelu?sequential_17/tcn_17/residual_block_2/conv1D_1/BiasAdd:output:0*
T0*-
_output_shapes
:?????????2;
9sequential_17/tcn_17/residual_block_2/activation_389/Relu
Dsequential_17/tcn_17/residual_block_2/spatial_dropout1d_195/IdentityIdentityGsequential_17/tcn_17/residual_block_2/activation_389/Relu:activations:0*
T0*-
_output_shapes
:?????????2F
Dsequential_17/tcn_17/residual_block_2/spatial_dropout1d_195/Identity
9sequential_17/tcn_17/residual_block_2/activation_390/ReluReluMsequential_17/tcn_17/residual_block_2/spatial_dropout1d_195/Identity:output:0*
T0*-
_output_shapes
:?????????2;
9sequential_17/tcn_17/residual_block_2/activation_390/Relu΅
/sequential_17/tcn_17/residual_block_2/add_2/addAddV2Gsequential_17/tcn_17/residual_block_1/activation_387/Relu:activations:0Gsequential_17/tcn_17/residual_block_2/activation_390/Relu:activations:0*
T0*-
_output_shapes
:?????????21
/sequential_17/tcn_17/residual_block_2/add_2/addλ
9sequential_17/tcn_17/residual_block_2/activation_391/ReluRelu3sequential_17/tcn_17/residual_block_2/add_2/add:z:0*
T0*-
_output_shapes
:?????????2;
9sequential_17/tcn_17/residual_block_2/activation_391/Relu
sequential_17/tcn_17/add_3/addAddV2Gsequential_17/tcn_17/residual_block_0/activation_382/Relu:activations:0Gsequential_17/tcn_17/residual_block_1/activation_386/Relu:activations:0*
T0*-
_output_shapes
:?????????2 
sequential_17/tcn_17/add_3/addς
 sequential_17/tcn_17/add_3/add_1AddV2"sequential_17/tcn_17/add_3/add:z:0Gsequential_17/tcn_17/residual_block_2/activation_390/Relu:activations:0*
T0*-
_output_shapes
:?????????2"
 sequential_17/tcn_17/add_3/add_1½
2sequential_17/tcn_17/lambda_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ????    24
2sequential_17/tcn_17/lambda_17/strided_slice/stackΑ
4sequential_17/tcn_17/lambda_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            26
4sequential_17/tcn_17/lambda_17/strided_slice/stack_1Α
4sequential_17/tcn_17/lambda_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         26
4sequential_17/tcn_17/lambda_17/strided_slice/stack_2Η
,sequential_17/tcn_17/lambda_17/strided_sliceStridedSlice$sequential_17/tcn_17/add_3/add_1:z:0;sequential_17/tcn_17/lambda_17/strided_slice/stack:output:0=sequential_17/tcn_17/lambda_17/strided_slice/stack_1:output:0=sequential_17/tcn_17/lambda_17/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2.
,sequential_17/tcn_17/lambda_17/strided_sliceΤ
,sequential_17/dense_34/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_34_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02.
,sequential_17/dense_34/MatMul/ReadVariableOpθ
sequential_17/dense_34/MatMulMatMul5sequential_17/tcn_17/lambda_17/strided_slice:output:04sequential_17/dense_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
sequential_17/dense_34/MatMul?
-sequential_17/dense_34/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_34_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_17/dense_34/BiasAdd/ReadVariableOpή
sequential_17/dense_34/BiasAddBiasAdd'sequential_17/dense_34/MatMul:product:05sequential_17/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2 
sequential_17/dense_34/BiasAdd
sequential_17/dense_34/ReluRelu'sequential_17/dense_34/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
sequential_17/dense_34/ReluΤ
,sequential_17/dense_35/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_35_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02.
,sequential_17/dense_35/MatMul/ReadVariableOpά
sequential_17/dense_35/MatMulMatMul)sequential_17/dense_34/Relu:activations:04sequential_17/dense_35/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
sequential_17/dense_35/MatMul?
-sequential_17/dense_35/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_35_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_17/dense_35/BiasAdd/ReadVariableOpή
sequential_17/dense_35/BiasAddBiasAdd'sequential_17/dense_35/MatMul:product:05sequential_17/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2 
sequential_17/dense_35/BiasAdd|
IdentityIdentity'sequential_17/dense_35/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:?????????:::::::::::::::::::Z V
,
_output_shapes
:?????????
&
_user_specified_nametcn_17_input"ΈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*»
serving_default§
J
tcn_17_input:
serving_default_tcn_17_input:0?????????=
dense_351
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ψ
Θ!
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
ͺ__call__
«_default_save_signature
+¬&call_and_return_all_conditional_losses"
_tf_keras_sequentialζ{"class_name": "Sequential", "name": "sequential_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "tcn_17_input"}}, {"class_name": "TCN", "config": {"name": "tcn_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 1]}, "dtype": "float32", "nb_filters": 128, "kernel_size": 64, "nb_stacks": 1, "dilations": [1, 2, 4], "padding": "causal", "use_skip_connections": true, "dropout_rate": 0.05, "return_sequences": false, "activation": "relu", "use_batch_norm": false, "use_layer_norm": false, "use_weight_norm": false, "kernel_initializer": "glorot_uniform"}}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "tcn_17_input"}}, {"class_name": "TCN", "config": {"name": "tcn_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 1]}, "dtype": "float32", "nb_filters": 128, "kernel_size": 64, "nb_stacks": 1, "dilations": [1, 2, 4], "padding": "causal", "use_skip_connections": true, "dropout_rate": 0.05, "return_sequences": false, "activation": "relu", "use_batch_norm": false, "use_layer_norm": false, "use_weight_norm": false, "kernel_initializer": "glorot_uniform"}}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "LogCosh", "config": {"reduction": "auto", "name": "log_cosh"}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Σ

	dilations
skip_connections
residual_blocks
layers_outputs
residual_block_0
residual_block_1
residual_block_2
slicer_layer
	variables
trainable_variables
regularization_losses
	keras_api
­__call__
+?&call_and_return_all_conditional_losses" 
_tf_keras_layer{"class_name": "TCN", "name": "tcn_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "tcn_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 1]}, "dtype": "float32", "nb_filters": 128, "kernel_size": 64, "nb_stacks": 1, "dilations": [1, 2, 4], "padding": "causal", "use_skip_connections": true, "dropout_rate": 0.05, "return_sequences": false, "activation": "relu", "use_batch_norm": false, "use_layer_norm": false, "use_weight_norm": false, "kernel_initializer": "glorot_uniform"}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 1]}}
χ

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
―__call__
+°&call_and_return_all_conditional_losses"Π
_tf_keras_layerΆ{"class_name": "Dense", "name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
ω

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
±__call__
+²&call_and_return_all_conditional_losses"?
_tf_keras_layerΈ{"class_name": "Dense", "name": "dense_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
»
"iter

#beta_1

$beta_2
	%decay
&learning_ratemmmm'm(m)m*m+m,m-m.m/m0m1m2m3m4mvvvv'v(v)v*v+v ,v‘-v’.v£/v€0v₯1v¦2v§3v¨4v©"
	optimizer
¦
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
14
15
16
17"
trackable_list_wrapper
¦
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
14
15
16
17"
trackable_list_wrapper
 "
trackable_list_wrapper
Ξ
5layer_regularization_losses
	variables
6layer_metrics
7metrics

8layers
9non_trainable_variables
trainable_variables
regularization_losses
ͺ__call__
«_default_save_signature
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
-
³serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper


:layers
;layers_outputs
<shape_match_conv
=final_activation
>conv1D_0
?activation_380
@spatial_dropout1d_190
Aconv1D_1
Bactivation_381
Cspatial_dropout1d_191
Dactivation_382
<matching_conv1D
=activation_383
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
΄__call__
+΅&call_and_return_all_conditional_losses"μ
_tf_keras_layer?{"class_name": "ResidualBlock", "name": "residual_block_0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 1]}}


Ilayers
Jlayers_outputs
Kshape_match_conv
Lfinal_activation
Mconv1D_0
Nactivation_384
Ospatial_dropout1d_192
Pconv1D_1
Qactivation_385
Rspatial_dropout1d_193
Sactivation_386
Kmatching_identity
Lactivation_387
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
Ά__call__
+·&call_and_return_all_conditional_losses"ξ
_tf_keras_layerΤ{"class_name": "ResidualBlock", "name": "residual_block_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 128]}}


Xlayers
Ylayers_outputs
Zshape_match_conv
[final_activation
\conv1D_0
]activation_388
^spatial_dropout1d_194
_conv1D_1
`activation_389
aspatial_dropout1d_195
bactivation_390
Zmatching_identity
[activation_391
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
Έ__call__
+Ή&call_and_return_all_conditional_losses"ξ
_tf_keras_layerΤ{"class_name": "ResidualBlock", "name": "residual_block_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 128]}}
ν

g	variables
htrainable_variables
iregularization_losses
j	keras_api
Ί__call__
+»&call_and_return_all_conditional_losses"ά	
_tf_keras_layerΒ	{"class_name": "Lambda", "name": "lambda_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda_17", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAFAAAAEwAAAHMYAAAAfABkAGQAhQKIAGoAZABkAIUCZgMZAFMAqQFO\nKQHaEm91dHB1dF9zbGljZV9pbmRleCkB2gJ0dKkB2gRzZWxmqQD6R0M6L1VzZXJzLzE2MTc3L0Fw\ncERhdGEvUm9hbWluZy9QeXRob24vUHl0aG9uMzgvc2l0ZS1wYWNrYWdlcy90Y24vdGNuLnB52gg8\nbGFtYmRhPiYBAADzAAAAAA==\n", null, {"class_name": "__tuple__", "items": [{"class_name": "TCN", "config": {"name": "tcn_17", "trainable": true, "batch_input_shape": [null, 512, 1], "dtype": "float32", "nb_filters": 128, "kernel_size": 64, "nb_stacks": 1, "dilations": [1, 2, 4], "padding": "causal", "use_skip_connections": true, "dropout_rate": 0.05, "return_sequences": false, "activation": "relu", "use_batch_norm": false, "use_layer_norm": false, "use_weight_norm": false, "kernel_initializer": "glorot_uniform"}}]}]}, "function_type": "lambda", "module": "tcn.tcn", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}

'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413"
trackable_list_wrapper

'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413"
trackable_list_wrapper
 "
trackable_list_wrapper
°
klayer_regularization_losses
	variables
llayer_metrics
mmetrics

nlayers
onon_trainable_variables
trainable_variables
regularization_losses
­__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_34/kernel
:2dense_34/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
player_regularization_losses
	variables
qlayer_metrics
rmetrics

slayers
tnon_trainable_variables
trainable_variables
regularization_losses
―__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_35/kernel
:2dense_35/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
ulayer_regularization_losses
	variables
vlayer_metrics
wmetrics

xlayers
ynon_trainable_variables
trainable_variables
 regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
>:<@2'tcn_17/residual_block_0/conv1D_0/kernel
4:22%tcn_17/residual_block_0/conv1D_0/bias
?:=@2'tcn_17/residual_block_0/conv1D_1/kernel
4:22%tcn_17/residual_block_0/conv1D_1/bias
E:C2.tcn_17/residual_block_0/matching_conv1D/kernel
;:92,tcn_17/residual_block_0/matching_conv1D/bias
?:=@2'tcn_17/residual_block_1/conv1D_0/kernel
4:22%tcn_17/residual_block_1/conv1D_0/bias
?:=@2'tcn_17/residual_block_1/conv1D_1/kernel
4:22%tcn_17/residual_block_1/conv1D_1/bias
?:=@2'tcn_17/residual_block_2/conv1D_0/kernel
4:22%tcn_17/residual_block_2/conv1D_0/bias
?:=@2'tcn_17/residual_block_2/conv1D_1/kernel
4:22%tcn_17/residual_block_2/conv1D_1/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
z0"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
Q
>0
?1
@2
A3
B4
C5
D6"
trackable_list_wrapper
 "
trackable_list_wrapper
ͺ	

+kernel
,bias
{	variables
|trainable_variables
}regularization_losses
~	keras_api
Ό__call__
+½&call_and_return_all_conditional_losses"
_tf_keras_layerι{"class_name": "Conv1D", "name": "matching_conv1D", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "matching_conv1D", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}}
ή
	variables
trainable_variables
regularization_losses
	keras_api
Ύ__call__
+Ώ&call_and_return_all_conditional_losses"Κ
_tf_keras_layer°{"class_name": "Activation", "name": "activation_383", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_383", "trainable": true, "dtype": "float32", "activation": "relu"}}
£	

'kernel
(bias
	variables
trainable_variables
regularization_losses
	keras_api
ΐ__call__
+Α&call_and_return_all_conditional_losses"ψ
_tf_keras_layerή{"class_name": "Conv1D", "name": "conv1D_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1D_0", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [64]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}}
ί
	variables
trainable_variables
regularization_losses
	keras_api
Β__call__
+Γ&call_and_return_all_conditional_losses"Κ
_tf_keras_layer°{"class_name": "Activation", "name": "activation_380", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_380", "trainable": true, "dtype": "float32", "activation": "relu"}}

	variables
trainable_variables
regularization_losses
	keras_api
Δ__call__
+Ε&call_and_return_all_conditional_losses"
_tf_keras_layerξ{"class_name": "SpatialDropout1D", "name": "spatial_dropout1d_190", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "spatial_dropout1d_190", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
₯	

)kernel
*bias
	variables
trainable_variables
regularization_losses
	keras_api
Ζ__call__
+Η&call_and_return_all_conditional_losses"ϊ
_tf_keras_layerΰ{"class_name": "Conv1D", "name": "conv1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1D_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [64]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}}
ί
	variables
trainable_variables
regularization_losses
	keras_api
Θ__call__
+Ι&call_and_return_all_conditional_losses"Κ
_tf_keras_layer°{"class_name": "Activation", "name": "activation_381", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_381", "trainable": true, "dtype": "float32", "activation": "relu"}}

	variables
trainable_variables
regularization_losses
	keras_api
Κ__call__
+Λ&call_and_return_all_conditional_losses"
_tf_keras_layerξ{"class_name": "SpatialDropout1D", "name": "spatial_dropout1d_191", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "spatial_dropout1d_191", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ί
	variables
trainable_variables
regularization_losses
	keras_api
Μ__call__
+Ν&call_and_return_all_conditional_losses"Κ
_tf_keras_layer°{"class_name": "Activation", "name": "activation_382", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_382", "trainable": true, "dtype": "float32", "activation": "relu"}}
J
'0
(1
)2
*3
+4
,5"
trackable_list_wrapper
J
'0
(1
)2
*3
+4
,5"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
 layer_regularization_losses
E	variables
 layer_metrics
‘metrics
’layers
£non_trainable_variables
Ftrainable_variables
Gregularization_losses
΄__call__
+΅&call_and_return_all_conditional_losses
'΅"call_and_return_conditional_losses"
_generic_user_object
Q
M0
N1
O2
P3
Q4
R5
S6"
trackable_list_wrapper
 "
trackable_list_wrapper
θ
€	variables
₯trainable_variables
¦regularization_losses
§	keras_api
Ξ__call__
+Ο&call_and_return_all_conditional_losses"Σ
_tf_keras_layerΉ{"class_name": "Lambda", "name": "matching_identity", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "matching_identity", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAABAAAAUwAAAHMEAAAAfABTAKkBTqkAKQHaAXhyAgAAAHICAAAA+kdD\nOi9Vc2Vycy8xNjE3Ny9BcHBEYXRhL1JvYW1pbmcvUHl0aG9uL1B5dGhvbjM4L3NpdGUtcGFja2Fn\nZXMvdGNuL3Rjbi5wedoIPGxhbWJkYT6AAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "tcn.tcn", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
ί
¨	variables
©trainable_variables
ͺregularization_losses
«	keras_api
Π__call__
+Ρ&call_and_return_all_conditional_losses"Κ
_tf_keras_layer°{"class_name": "Activation", "name": "activation_387", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_387", "trainable": true, "dtype": "float32", "activation": "relu"}}
₯	

-kernel
.bias
¬	variables
­trainable_variables
?regularization_losses
―	keras_api
?__call__
+Σ&call_and_return_all_conditional_losses"ϊ
_tf_keras_layerΰ{"class_name": "Conv1D", "name": "conv1D_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1D_0", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [64]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}}
ί
°	variables
±trainable_variables
²regularization_losses
³	keras_api
Τ__call__
+Υ&call_and_return_all_conditional_losses"Κ
_tf_keras_layer°{"class_name": "Activation", "name": "activation_384", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_384", "trainable": true, "dtype": "float32", "activation": "relu"}}

΄	variables
΅trainable_variables
Άregularization_losses
·	keras_api
Φ__call__
+Χ&call_and_return_all_conditional_losses"
_tf_keras_layerξ{"class_name": "SpatialDropout1D", "name": "spatial_dropout1d_192", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "spatial_dropout1d_192", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
₯	

/kernel
0bias
Έ	variables
Ήtrainable_variables
Ίregularization_losses
»	keras_api
Ψ__call__
+Ω&call_and_return_all_conditional_losses"ϊ
_tf_keras_layerΰ{"class_name": "Conv1D", "name": "conv1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1D_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [64]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [2]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}}
ί
Ό	variables
½trainable_variables
Ύregularization_losses
Ώ	keras_api
Ϊ__call__
+Ϋ&call_and_return_all_conditional_losses"Κ
_tf_keras_layer°{"class_name": "Activation", "name": "activation_385", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_385", "trainable": true, "dtype": "float32", "activation": "relu"}}

ΐ	variables
Αtrainable_variables
Βregularization_losses
Γ	keras_api
ά__call__
+έ&call_and_return_all_conditional_losses"
_tf_keras_layerξ{"class_name": "SpatialDropout1D", "name": "spatial_dropout1d_193", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "spatial_dropout1d_193", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ί
Δ	variables
Εtrainable_variables
Ζregularization_losses
Η	keras_api
ή__call__
+ί&call_and_return_all_conditional_losses"Κ
_tf_keras_layer°{"class_name": "Activation", "name": "activation_386", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_386", "trainable": true, "dtype": "float32", "activation": "relu"}}
<
-0
.1
/2
03"
trackable_list_wrapper
<
-0
.1
/2
03"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
 Θlayer_regularization_losses
T	variables
Ιlayer_metrics
Κmetrics
Λlayers
Μnon_trainable_variables
Utrainable_variables
Vregularization_losses
Ά__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
Q
\0
]1
^2
_3
`4
a5
b6"
trackable_list_wrapper
 "
trackable_list_wrapper
θ
Ν	variables
Ξtrainable_variables
Οregularization_losses
Π	keras_api
ΰ__call__
+α&call_and_return_all_conditional_losses"Σ
_tf_keras_layerΉ{"class_name": "Lambda", "name": "matching_identity", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "matching_identity", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAABAAAAUwAAAHMEAAAAfABTAKkBTqkAKQHaAXhyAgAAAHICAAAA+kdD\nOi9Vc2Vycy8xNjE3Ny9BcHBEYXRhL1JvYW1pbmcvUHl0aG9uL1B5dGhvbjM4L3NpdGUtcGFja2Fn\nZXMvdGNuL3Rjbi5wedoIPGxhbWJkYT6AAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "tcn.tcn", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
ί
Ρ	variables
?trainable_variables
Σregularization_losses
Τ	keras_api
β__call__
+γ&call_and_return_all_conditional_losses"Κ
_tf_keras_layer°{"class_name": "Activation", "name": "activation_391", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_391", "trainable": true, "dtype": "float32", "activation": "relu"}}
₯	

1kernel
2bias
Υ	variables
Φtrainable_variables
Χregularization_losses
Ψ	keras_api
δ__call__
+ε&call_and_return_all_conditional_losses"ϊ
_tf_keras_layerΰ{"class_name": "Conv1D", "name": "conv1D_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1D_0", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [64]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [4]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}}
ί
Ω	variables
Ϊtrainable_variables
Ϋregularization_losses
ά	keras_api
ζ__call__
+η&call_and_return_all_conditional_losses"Κ
_tf_keras_layer°{"class_name": "Activation", "name": "activation_388", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_388", "trainable": true, "dtype": "float32", "activation": "relu"}}

έ	variables
ήtrainable_variables
ίregularization_losses
ΰ	keras_api
θ__call__
+ι&call_and_return_all_conditional_losses"
_tf_keras_layerξ{"class_name": "SpatialDropout1D", "name": "spatial_dropout1d_194", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "spatial_dropout1d_194", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
₯	

3kernel
4bias
α	variables
βtrainable_variables
γregularization_losses
δ	keras_api
κ__call__
+λ&call_and_return_all_conditional_losses"ϊ
_tf_keras_layerΰ{"class_name": "Conv1D", "name": "conv1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1D_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [64]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "causal", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [4]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}}
ί
ε	variables
ζtrainable_variables
ηregularization_losses
θ	keras_api
μ__call__
+ν&call_and_return_all_conditional_losses"Κ
_tf_keras_layer°{"class_name": "Activation", "name": "activation_389", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_389", "trainable": true, "dtype": "float32", "activation": "relu"}}

ι	variables
κtrainable_variables
λregularization_losses
μ	keras_api
ξ__call__
+ο&call_and_return_all_conditional_losses"
_tf_keras_layerξ{"class_name": "SpatialDropout1D", "name": "spatial_dropout1d_195", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "spatial_dropout1d_195", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ί
ν	variables
ξtrainable_variables
οregularization_losses
π	keras_api
π__call__
+ρ&call_and_return_all_conditional_losses"Κ
_tf_keras_layer°{"class_name": "Activation", "name": "activation_390", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_390", "trainable": true, "dtype": "float32", "activation": "relu"}}
<
10
21
32
43"
trackable_list_wrapper
<
10
21
32
43"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
 ρlayer_regularization_losses
c	variables
ςlayer_metrics
σmetrics
τlayers
υnon_trainable_variables
dtrainable_variables
eregularization_losses
Έ__call__
+Ή&call_and_return_all_conditional_losses
'Ή"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
 φlayer_regularization_losses
g	variables
χlayer_metrics
ψmetrics
ωlayers
ϊnon_trainable_variables
htrainable_variables
iregularization_losses
Ί__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ώ

ϋtotal

όcount
ύ	variables
ώ	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
 ?layer_regularization_losses
{	variables
layer_metrics
metrics
layers
non_trainable_variables
|trainable_variables
}regularization_losses
Ό__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
·
 layer_regularization_losses
	variables
layer_metrics
metrics
layers
non_trainable_variables
trainable_variables
regularization_losses
Ύ__call__
+Ώ&call_and_return_all_conditional_losses
'Ώ"call_and_return_conditional_losses"
_generic_user_object
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 layer_regularization_losses
	variables
layer_metrics
metrics
layers
non_trainable_variables
trainable_variables
regularization_losses
ΐ__call__
+Α&call_and_return_all_conditional_losses
'Α"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 layer_regularization_losses
	variables
layer_metrics
metrics
layers
non_trainable_variables
trainable_variables
regularization_losses
Β__call__
+Γ&call_and_return_all_conditional_losses
'Γ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 layer_regularization_losses
	variables
layer_metrics
metrics
layers
non_trainable_variables
trainable_variables
regularization_losses
Δ__call__
+Ε&call_and_return_all_conditional_losses
'Ε"call_and_return_conditional_losses"
_generic_user_object
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 layer_regularization_losses
	variables
layer_metrics
metrics
layers
non_trainable_variables
trainable_variables
regularization_losses
Ζ__call__
+Η&call_and_return_all_conditional_losses
'Η"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 layer_regularization_losses
	variables
layer_metrics
metrics
 layers
‘non_trainable_variables
trainable_variables
regularization_losses
Θ__call__
+Ι&call_and_return_all_conditional_losses
'Ι"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 ’layer_regularization_losses
	variables
£layer_metrics
€metrics
₯layers
¦non_trainable_variables
trainable_variables
regularization_losses
Κ__call__
+Λ&call_and_return_all_conditional_losses
'Λ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 §layer_regularization_losses
	variables
¨layer_metrics
©metrics
ͺlayers
«non_trainable_variables
trainable_variables
regularization_losses
Μ__call__
+Ν&call_and_return_all_conditional_losses
'Ν"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
_
>0
?1
@2
A3
B4
C5
D6
<7
=8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 ¬layer_regularization_losses
€	variables
­layer_metrics
?metrics
―layers
°non_trainable_variables
₯trainable_variables
¦regularization_losses
Ξ__call__
+Ο&call_and_return_all_conditional_losses
'Ο"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 ±layer_regularization_losses
¨	variables
²layer_metrics
³metrics
΄layers
΅non_trainable_variables
©trainable_variables
ͺregularization_losses
Π__call__
+Ρ&call_and_return_all_conditional_losses
'Ρ"call_and_return_conditional_losses"
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
Έ
 Άlayer_regularization_losses
¬	variables
·layer_metrics
Έmetrics
Ήlayers
Ίnon_trainable_variables
­trainable_variables
?regularization_losses
?__call__
+Σ&call_and_return_all_conditional_losses
'Σ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 »layer_regularization_losses
°	variables
Όlayer_metrics
½metrics
Ύlayers
Ώnon_trainable_variables
±trainable_variables
²regularization_losses
Τ__call__
+Υ&call_and_return_all_conditional_losses
'Υ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 ΐlayer_regularization_losses
΄	variables
Αlayer_metrics
Βmetrics
Γlayers
Δnon_trainable_variables
΅trainable_variables
Άregularization_losses
Φ__call__
+Χ&call_and_return_all_conditional_losses
'Χ"call_and_return_conditional_losses"
_generic_user_object
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 Εlayer_regularization_losses
Έ	variables
Ζlayer_metrics
Ηmetrics
Θlayers
Ιnon_trainable_variables
Ήtrainable_variables
Ίregularization_losses
Ψ__call__
+Ω&call_and_return_all_conditional_losses
'Ω"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 Κlayer_regularization_losses
Ό	variables
Λlayer_metrics
Μmetrics
Νlayers
Ξnon_trainable_variables
½trainable_variables
Ύregularization_losses
Ϊ__call__
+Ϋ&call_and_return_all_conditional_losses
'Ϋ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 Οlayer_regularization_losses
ΐ	variables
Πlayer_metrics
Ρmetrics
?layers
Σnon_trainable_variables
Αtrainable_variables
Βregularization_losses
ά__call__
+έ&call_and_return_all_conditional_losses
'έ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 Τlayer_regularization_losses
Δ	variables
Υlayer_metrics
Φmetrics
Χlayers
Ψnon_trainable_variables
Εtrainable_variables
Ζregularization_losses
ή__call__
+ί&call_and_return_all_conditional_losses
'ί"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
_
M0
N1
O2
P3
Q4
R5
S6
K7
L8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 Ωlayer_regularization_losses
Ν	variables
Ϊlayer_metrics
Ϋmetrics
άlayers
έnon_trainable_variables
Ξtrainable_variables
Οregularization_losses
ΰ__call__
+α&call_and_return_all_conditional_losses
'α"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 ήlayer_regularization_losses
Ρ	variables
ίlayer_metrics
ΰmetrics
αlayers
βnon_trainable_variables
?trainable_variables
Σregularization_losses
β__call__
+γ&call_and_return_all_conditional_losses
'γ"call_and_return_conditional_losses"
_generic_user_object
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 γlayer_regularization_losses
Υ	variables
δlayer_metrics
εmetrics
ζlayers
ηnon_trainable_variables
Φtrainable_variables
Χregularization_losses
δ__call__
+ε&call_and_return_all_conditional_losses
'ε"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 θlayer_regularization_losses
Ω	variables
ιlayer_metrics
κmetrics
λlayers
μnon_trainable_variables
Ϊtrainable_variables
Ϋregularization_losses
ζ__call__
+η&call_and_return_all_conditional_losses
'η"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 νlayer_regularization_losses
έ	variables
ξlayer_metrics
οmetrics
πlayers
ρnon_trainable_variables
ήtrainable_variables
ίregularization_losses
θ__call__
+ι&call_and_return_all_conditional_losses
'ι"call_and_return_conditional_losses"
_generic_user_object
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 ςlayer_regularization_losses
α	variables
σlayer_metrics
τmetrics
υlayers
φnon_trainable_variables
βtrainable_variables
γregularization_losses
κ__call__
+λ&call_and_return_all_conditional_losses
'λ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 χlayer_regularization_losses
ε	variables
ψlayer_metrics
ωmetrics
ϊlayers
ϋnon_trainable_variables
ζtrainable_variables
ηregularization_losses
μ__call__
+ν&call_and_return_all_conditional_losses
'ν"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 όlayer_regularization_losses
ι	variables
ύlayer_metrics
ώmetrics
?layers
non_trainable_variables
κtrainable_variables
λregularization_losses
ξ__call__
+ο&call_and_return_all_conditional_losses
'ο"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 layer_regularization_losses
ν	variables
layer_metrics
metrics
layers
non_trainable_variables
ξtrainable_variables
οregularization_losses
π__call__
+ρ&call_and_return_all_conditional_losses
'ρ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
_
\0
]1
^2
_3
`4
a5
b6
Z7
[8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
ϋ0
ό1"
trackable_list_wrapper
.
ύ	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(:&
2Adam/dense_34/kernel/m
!:2Adam/dense_34/bias/m
(:&
2Adam/dense_35/kernel/m
!:2Adam/dense_35/bias/m
C:A@2.Adam/tcn_17/residual_block_0/conv1D_0/kernel/m
9:72,Adam/tcn_17/residual_block_0/conv1D_0/bias/m
D:B@2.Adam/tcn_17/residual_block_0/conv1D_1/kernel/m
9:72,Adam/tcn_17/residual_block_0/conv1D_1/bias/m
J:H25Adam/tcn_17/residual_block_0/matching_conv1D/kernel/m
@:>23Adam/tcn_17/residual_block_0/matching_conv1D/bias/m
D:B@2.Adam/tcn_17/residual_block_1/conv1D_0/kernel/m
9:72,Adam/tcn_17/residual_block_1/conv1D_0/bias/m
D:B@2.Adam/tcn_17/residual_block_1/conv1D_1/kernel/m
9:72,Adam/tcn_17/residual_block_1/conv1D_1/bias/m
D:B@2.Adam/tcn_17/residual_block_2/conv1D_0/kernel/m
9:72,Adam/tcn_17/residual_block_2/conv1D_0/bias/m
D:B@2.Adam/tcn_17/residual_block_2/conv1D_1/kernel/m
9:72,Adam/tcn_17/residual_block_2/conv1D_1/bias/m
(:&
2Adam/dense_34/kernel/v
!:2Adam/dense_34/bias/v
(:&
2Adam/dense_35/kernel/v
!:2Adam/dense_35/bias/v
C:A@2.Adam/tcn_17/residual_block_0/conv1D_0/kernel/v
9:72,Adam/tcn_17/residual_block_0/conv1D_0/bias/v
D:B@2.Adam/tcn_17/residual_block_0/conv1D_1/kernel/v
9:72,Adam/tcn_17/residual_block_0/conv1D_1/bias/v
J:H25Adam/tcn_17/residual_block_0/matching_conv1D/kernel/v
@:>23Adam/tcn_17/residual_block_0/matching_conv1D/bias/v
D:B@2.Adam/tcn_17/residual_block_1/conv1D_0/kernel/v
9:72,Adam/tcn_17/residual_block_1/conv1D_0/bias/v
D:B@2.Adam/tcn_17/residual_block_1/conv1D_1/kernel/v
9:72,Adam/tcn_17/residual_block_1/conv1D_1/bias/v
D:B@2.Adam/tcn_17/residual_block_2/conv1D_0/kernel/v
9:72,Adam/tcn_17/residual_block_2/conv1D_0/bias/v
D:B@2.Adam/tcn_17/residual_block_2/conv1D_1/kernel/v
9:72,Adam/tcn_17/residual_block_2/conv1D_1/bias/v
2
/__inference_sequential_17_layer_call_fn_1156659
/__inference_sequential_17_layer_call_fn_1156027
/__inference_sequential_17_layer_call_fn_1156111
/__inference_sequential_17_layer_call_fn_1156700ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
κ2η
"__inference__wrapped_model_1154918ΐ
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *0’-
+(
tcn_17_input?????????
φ2σ
J__inference_sequential_17_layer_call_and_return_conditional_losses_1156618
J__inference_sequential_17_layer_call_and_return_conditional_losses_1155942
J__inference_sequential_17_layer_call_and_return_conditional_losses_1156441
J__inference_sequential_17_layer_call_and_return_conditional_losses_1155899ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
(__inference_tcn_17_layer_call_fn_1157163
(__inference_tcn_17_layer_call_fn_1157196΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Δ2Α
C__inference_tcn_17_layer_call_and_return_conditional_losses_1157130
C__inference_tcn_17_layer_call_and_return_conditional_losses_1156966΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Τ2Ρ
*__inference_dense_34_layer_call_fn_1157216’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_dense_34_layer_call_and_return_conditional_losses_1157207’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Τ2Ρ
*__inference_dense_35_layer_call_fn_1157235’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_dense_35_layer_call_and_return_conditional_losses_1157226’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
9B7
%__inference_signature_wrapper_1156162tcn_17_input
Ί2·΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ί2·΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ί2·΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ί2·΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ί2·΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ί2·΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ζ2Γΐ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ζ2Γΐ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¬2©
7__inference_spatial_dropout1d_190_layer_call_fn_1157272
7__inference_spatial_dropout1d_190_layer_call_fn_1157267΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
β2ί
R__inference_spatial_dropout1d_190_layer_call_and_return_conditional_losses_1157262
R__inference_spatial_dropout1d_190_layer_call_and_return_conditional_losses_1157257΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¬2©
7__inference_spatial_dropout1d_191_layer_call_fn_1157304
7__inference_spatial_dropout1d_191_layer_call_fn_1157309΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
β2ί
R__inference_spatial_dropout1d_191_layer_call_and_return_conditional_losses_1157299
R__inference_spatial_dropout1d_191_layer_call_and_return_conditional_losses_1157294΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ζ2Γΐ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ζ2Γΐ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¬2©
7__inference_spatial_dropout1d_192_layer_call_fn_1157346
7__inference_spatial_dropout1d_192_layer_call_fn_1157341΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
β2ί
R__inference_spatial_dropout1d_192_layer_call_and_return_conditional_losses_1157331
R__inference_spatial_dropout1d_192_layer_call_and_return_conditional_losses_1157336΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¬2©
7__inference_spatial_dropout1d_193_layer_call_fn_1157378
7__inference_spatial_dropout1d_193_layer_call_fn_1157383΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
β2ί
R__inference_spatial_dropout1d_193_layer_call_and_return_conditional_losses_1157373
R__inference_spatial_dropout1d_193_layer_call_and_return_conditional_losses_1157368΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ζ2Γΐ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ζ2Γΐ
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¬2©
7__inference_spatial_dropout1d_194_layer_call_fn_1157415
7__inference_spatial_dropout1d_194_layer_call_fn_1157420΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
β2ί
R__inference_spatial_dropout1d_194_layer_call_and_return_conditional_losses_1157410
R__inference_spatial_dropout1d_194_layer_call_and_return_conditional_losses_1157405΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¬2©
7__inference_spatial_dropout1d_195_layer_call_fn_1157457
7__inference_spatial_dropout1d_195_layer_call_fn_1157452΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
β2ί
R__inference_spatial_dropout1d_195_layer_call_and_return_conditional_losses_1157442
R__inference_spatial_dropout1d_195_layer_call_and_return_conditional_losses_1157447΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ­
"__inference__wrapped_model_1154918'()*+,-./01234:’7
0’-
+(
tcn_17_input?????????
ͺ "4ͺ1
/
dense_35# 
dense_35?????????§
E__inference_dense_34_layer_call_and_return_conditional_losses_1157207^0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 
*__inference_dense_34_layer_call_fn_1157216Q0’-
&’#
!
inputs?????????
ͺ "?????????§
E__inference_dense_35_layer_call_and_return_conditional_losses_1157226^0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 
*__inference_dense_35_layer_call_fn_1157235Q0’-
&’#
!
inputs?????????
ͺ "?????????Ο
J__inference_sequential_17_layer_call_and_return_conditional_losses_1155899'()*+,-./01234B’?
8’5
+(
tcn_17_input?????????
p

 
ͺ "&’#

0?????????
 Ο
J__inference_sequential_17_layer_call_and_return_conditional_losses_1155942'()*+,-./01234B’?
8’5
+(
tcn_17_input?????????
p 

 
ͺ "&’#

0?????????
 Θ
J__inference_sequential_17_layer_call_and_return_conditional_losses_1156441z'()*+,-./01234<’9
2’/
%"
inputs?????????
p

 
ͺ "&’#

0?????????
 Θ
J__inference_sequential_17_layer_call_and_return_conditional_losses_1156618z'()*+,-./01234<’9
2’/
%"
inputs?????????
p 

 
ͺ "&’#

0?????????
 ¦
/__inference_sequential_17_layer_call_fn_1156027s'()*+,-./01234B’?
8’5
+(
tcn_17_input?????????
p

 
ͺ "?????????¦
/__inference_sequential_17_layer_call_fn_1156111s'()*+,-./01234B’?
8’5
+(
tcn_17_input?????????
p 

 
ͺ "????????? 
/__inference_sequential_17_layer_call_fn_1156659m'()*+,-./01234<’9
2’/
%"
inputs?????????
p

 
ͺ "????????? 
/__inference_sequential_17_layer_call_fn_1156700m'()*+,-./01234<’9
2’/
%"
inputs?????????
p 

 
ͺ "?????????ΐ
%__inference_signature_wrapper_1156162'()*+,-./01234J’G
’ 
@ͺ=
;
tcn_17_input+(
tcn_17_input?????????"4ͺ1
/
dense_35# 
dense_35?????????ί
R__inference_spatial_dropout1d_190_layer_call_and_return_conditional_losses_1157257I’F
?’<
63
inputs'???????????????????????????
p
ͺ ";’8
1.
0'???????????????????????????
 ί
R__inference_spatial_dropout1d_190_layer_call_and_return_conditional_losses_1157262I’F
?’<
63
inputs'???????????????????????????
p 
ͺ ";’8
1.
0'???????????????????????????
 Ά
7__inference_spatial_dropout1d_190_layer_call_fn_1157267{I’F
?’<
63
inputs'???????????????????????????
p
ͺ ".+'???????????????????????????Ά
7__inference_spatial_dropout1d_190_layer_call_fn_1157272{I’F
?’<
63
inputs'???????????????????????????
p 
ͺ ".+'???????????????????????????ί
R__inference_spatial_dropout1d_191_layer_call_and_return_conditional_losses_1157294I’F
?’<
63
inputs'???????????????????????????
p
ͺ ";’8
1.
0'???????????????????????????
 ί
R__inference_spatial_dropout1d_191_layer_call_and_return_conditional_losses_1157299I’F
?’<
63
inputs'???????????????????????????
p 
ͺ ";’8
1.
0'???????????????????????????
 Ά
7__inference_spatial_dropout1d_191_layer_call_fn_1157304{I’F
?’<
63
inputs'???????????????????????????
p
ͺ ".+'???????????????????????????Ά
7__inference_spatial_dropout1d_191_layer_call_fn_1157309{I’F
?’<
63
inputs'???????????????????????????
p 
ͺ ".+'???????????????????????????ί
R__inference_spatial_dropout1d_192_layer_call_and_return_conditional_losses_1157331I’F
?’<
63
inputs'???????????????????????????
p
ͺ ";’8
1.
0'???????????????????????????
 ί
R__inference_spatial_dropout1d_192_layer_call_and_return_conditional_losses_1157336I’F
?’<
63
inputs'???????????????????????????
p 
ͺ ";’8
1.
0'???????????????????????????
 Ά
7__inference_spatial_dropout1d_192_layer_call_fn_1157341{I’F
?’<
63
inputs'???????????????????????????
p
ͺ ".+'???????????????????????????Ά
7__inference_spatial_dropout1d_192_layer_call_fn_1157346{I’F
?’<
63
inputs'???????????????????????????
p 
ͺ ".+'???????????????????????????ί
R__inference_spatial_dropout1d_193_layer_call_and_return_conditional_losses_1157368I’F
?’<
63
inputs'???????????????????????????
p
ͺ ";’8
1.
0'???????????????????????????
 ί
R__inference_spatial_dropout1d_193_layer_call_and_return_conditional_losses_1157373I’F
?’<
63
inputs'???????????????????????????
p 
ͺ ";’8
1.
0'???????????????????????????
 Ά
7__inference_spatial_dropout1d_193_layer_call_fn_1157378{I’F
?’<
63
inputs'???????????????????????????
p
ͺ ".+'???????????????????????????Ά
7__inference_spatial_dropout1d_193_layer_call_fn_1157383{I’F
?’<
63
inputs'???????????????????????????
p 
ͺ ".+'???????????????????????????ί
R__inference_spatial_dropout1d_194_layer_call_and_return_conditional_losses_1157405I’F
?’<
63
inputs'???????????????????????????
p
ͺ ";’8
1.
0'???????????????????????????
 ί
R__inference_spatial_dropout1d_194_layer_call_and_return_conditional_losses_1157410I’F
?’<
63
inputs'???????????????????????????
p 
ͺ ";’8
1.
0'???????????????????????????
 Ά
7__inference_spatial_dropout1d_194_layer_call_fn_1157415{I’F
?’<
63
inputs'???????????????????????????
p
ͺ ".+'???????????????????????????Ά
7__inference_spatial_dropout1d_194_layer_call_fn_1157420{I’F
?’<
63
inputs'???????????????????????????
p 
ͺ ".+'???????????????????????????ί
R__inference_spatial_dropout1d_195_layer_call_and_return_conditional_losses_1157442I’F
?’<
63
inputs'???????????????????????????
p
ͺ ";’8
1.
0'???????????????????????????
 ί
R__inference_spatial_dropout1d_195_layer_call_and_return_conditional_losses_1157447I’F
?’<
63
inputs'???????????????????????????
p 
ͺ ";’8
1.
0'???????????????????????????
 Ά
7__inference_spatial_dropout1d_195_layer_call_fn_1157452{I’F
?’<
63
inputs'???????????????????????????
p
ͺ ".+'???????????????????????????Ά
7__inference_spatial_dropout1d_195_layer_call_fn_1157457{I’F
?’<
63
inputs'???????????????????????????
p 
ͺ ".+'???????????????????????????Ή
C__inference_tcn_17_layer_call_and_return_conditional_losses_1156966r'()*+,-./012348’5
.’+
%"
inputs?????????
p
ͺ "&’#

0?????????
 Ή
C__inference_tcn_17_layer_call_and_return_conditional_losses_1157130r'()*+,-./012348’5
.’+
%"
inputs?????????
p 
ͺ "&’#

0?????????
 
(__inference_tcn_17_layer_call_fn_1157163e'()*+,-./012348’5
.’+
%"
inputs?????????
p
ͺ "?????????
(__inference_tcn_17_layer_call_fn_1157196e'()*+,-./012348’5
.’+
%"
inputs?????????
p 
ͺ "?????????