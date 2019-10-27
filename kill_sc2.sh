#!/bin/bash
sudo kill -9 $(ps aux | grep 'StarCraftII' | awk '{print $2}')
