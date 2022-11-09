#!/bin/bash

TARGET_CONF_PATH="/etc/gala-anteater/config/"

# copy config file to target path
if [ ! -d "${TARGET_CONF_PATH}" ]; then
    mkdir -p ${TARGET_CONF_PATH}
fi

for conf_file in `ls config/`; do
    if [ ! -f "${TARGET_CONF_PATH}${conf_file}" ]; then
        cp config/${conf_file} ${TARGET_CONF_PATH}
    fi
done

exec "$@"
