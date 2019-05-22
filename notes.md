# Golden features on IDS2017

## Using LARS on a subsample of the dataset
Chosen features are: 
```
['destinationport'
 'flowbytes/s'
  'fwdiatmean' 
  'bwdiattotal'
 'bwdheaderlength' 
 'pshflagcount' 
 'urgflagcount' 
 'down/upratio'
 'init_win_bytes_forward' 
 'init_win_bytes_backward']

```

## Decision Tree (max_depth=3) feature importances on subsample of dataset
In order most to least important
```
['pshflagcount'
 'init_win_bytes_backward' 
 'avgfwdsegmentsize'
 'packetlengthmean' 
 'subflowfwdbytes' 
 'fwdiatmean' 
 'fwdheaderlength'
 'fwdpshflags' 
 'bwdiatmean' 
 'bwdiattotal']

```
with accuracy `0.9925`. 

## Linear Clf on subsample of dataset
In order most to least important

``` 
['subflowbwdbytes' 'act_data_pkt_fwd' 'bwdheaderlength' 'flowduration'
 'fwdiattotal' 'packetlengthmean' 'bwdpacketlengthmean'
 'avgbwdsegmentsize' 'fwdiatmean' 'pshflagcount']

``` 
with accuracy `0.9915`. 

# Golden features on KDD
## Using LARS on a subsample of the dataset
Chosen features, without any order: 
``` 
['wrong_fragment' 'srv_rerror_rate' 'same_srv_rate' 'diff_srv_rate'
 'dst_host_srv_count' 'dst_host_same_srv_rate'
 'dst_host_same_src_port_rate' 'dst_host_srv_diff_host_rate'
 'dst_host_serror_rate' 'dst_host_srv_rerror_rate']

``` 

## Dcision Tree trained on global subset 
``` 

['serror_rate' 
'same_srv_rate' 
'rerror_rate' 
'count'
 'dst_host_rerror_rate' 
 'logged_in' 
 'dst_host_srv_rerror_rate'
 'root_shell' 
 'num_file_creations' 'num_root']

``` 

## Linear Clf trained on global subset

``` 
['num_compromised' 
'num_root' 
'same_srv_rate'
 'srv_serror_rate'
 'dst_host_srv_serror_rate'
  'srv_rerror_rate'
   'rerror_rate' 
   'serror_rate'
 'wrong_fragment'
  'dst_host_srv_count']
``` 

## DT and Linear - common features 
serror_rate
same_srv_rate
rerror_rate
num_root
dst_host_srv_rerror_rate


## RandomForest on DB 
```
src_bytes 0.11561479014169512
same_srv_rate 0.09236295389801726
dst_host_same_srv_rate 0.06625651262512335
diff_srv_rate 0.05915526176845775
dst_bytes 0.05357894933104285
count 0.04094825130295683
serror_rate -0.038385065350561985
dst_host_serror_rate -0.03456708687080278
dst_host_srv_rerror_rate -0.031514398597607206
dst_host_srv_serror_rate -0.03128614516468414
```
