#!/bin/bash

TARGET_CONF_PATH="/etc/gala-anteater/config/"
TARGET_MODULE_PATH="/etc/gala-anteater/module/"

# copy config file to target path``
if [ ! -d "${TARGET_CONF_PATH}" ]; then
    mkdir -p ${TARGET_CONF_PATH}
fi

if [ ! -d "${TARGET_MODULE_PATH}" ]; then
    mkdir -p ${TARGET_MODULE_PATH}
fi

for file in `ls config/`; do
    if [ ! -f "${TARGET_CONF_PATH}${file}" ]; then
        cp -r config/${file} ${TARGET_CONF_PATH}
    fi
done

# remove "/etc/gala-anteater/config/module" folder
if [ -d "${TARGET_CONF_PATH}module" ]; then
    rm -rf ${TARGET_CONF_PATH}module
fi

# copy module config to TARGET_MODULE_PATH folder
for file in `ls config/module/`; do
    if [ ! -f "${TARGET_MODULE_PATH}${file}" ]; then
        cp -r config/module/${file} ${TARGET_MODULE_PATH}
    fi
done

exec "$@"
