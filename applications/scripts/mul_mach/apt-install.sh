#!/bin/bash
set -x


HOSTFILE='/home/cluster.conf'

for hostname in `cat ${HOSTFILE} | awk -F ' ' '{print $1}'`; do
  ssh $hostname "apt-get update 2>&1" &
done
wait

apt-get install pdsh

for hostname in `cat ${HOSTFILE} | awk -F ' ' '{print $1}'`; do
  ssh $hostname "apt-get install ninja-build 2>&1" &
done
wait
