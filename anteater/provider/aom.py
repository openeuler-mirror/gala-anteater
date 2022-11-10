#!/usr/bin/python3
# ******************************************************************************
# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# gala-anteater is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************************/
"""
Time:
Author:
Description: The implementation of AomAdapter client to fetch time series data.
"""

import time
from abc import ABCMeta, abstractmethod
from multiprocessing import AuthenticationError

import requests

from anteater.config import AomConfig
from anteater.provider.base import TimeSeriesProvider
from anteater.utils.log import logger


class AomAuth(metaclass=ABCMeta):
    @abstractmethod
    def set_auth_info(self, headers: dict):
        pass


class AppCodeAuth(AomAuth):
    def __init__(self, app_code: str):
        self._app_code = app_code

    def set_auth_info(self, headers: dict):
        headers['X-Apig-AppCode'] = self._app_code


class TokenAuth(AomAuth):
    def __init__(self, auth_info: dict):
        self._iam_user_name = auth_info.get('iam_user_name')  # iam_user_name
        self._iam_password = auth_info.get('iam_password')  # iam_password
        self._iam_domain = auth_info.get('iam_domain')  # iam_domain
        self._iam_server = auth_info.get('iam_server')  # iam_server
        self._verify = auth_info.get('ssl_verify')  # verify

        self._token: str = ''
        self._expires_at: int = 0
        # token 到期前 1 小时更新 token
        self._expire_duration: int = 3600

        self._token_api = '/v3/auth/tokens'

    @property
    def token(self):
        if not self._token or self.is_token_expired():
            self.update_token()
        return self._token

    def set_auth_info(self, headers: dict):
        headers['X-Auth-Token'] = self.token

    def update_token(self):
        headers = {
            'Content-Type': 'application/json;charset=utf8'
        }
        params = {
            'nocatalog': 'true'
        }
        body = {
            'auth': {
                'identity': {
                    'methods': ['password'],
                    'password': {
                        'user': {
                            'domain': {
                                'name': self._iam_domain
                            },
                            'name': self._iam_user_name,
                            'password': self._iam_password
                        }
                    }
                },
                'scope': {
                    'domain': {
                        'name': self._iam_domain
                    }
                }
            }
        }
        url = self._iam_server + self._token_api
        try:
            resp = requests.post(url, json=body, headers=headers, params=params, verify=self._verify)
        except requests.RequestException as ex:
            logger.error(ex)
            return
        try:
            resp_body = resp.json()
        except requests.RequestException as ex:
            logger.error(ex)
            return
        if resp.status_code != 201:
            logger.error('Failed to request {}, error is {}'.format(url, resp_body))
            return
        expires_at = resp_body.get('token', {}).get('expires_at')
        if not self._transfer_expire_time(expires_at):
            logger.error('Can not transfer expire time: {}'.format(expires_at))
            return
        self._token = resp.headers.get('X-Subject-Token')

    def is_token_expired(self) -> bool:
        return int(time.time()) + self._expire_duration > self._expires_at

    def _transfer_expire_time(self, s_time: str) -> bool:
        try:
            expires_at_arr = time.strptime(s_time, '%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError as ex:
            logger.error(ex)
            return False
        self._expires_at = int(time.mktime(expires_at_arr))
        return True


class AomAdapter(TimeSeriesProvider):
    """The AomAdapter client to consume time series data"""
    def __init__(self, aom_server: str, project_id: str, aom_auth: AomAuth):
        range_url = aom_server + f"/v1/{project_id}/aom/api/v1/query_range"
        self._aom_auth = aom_auth
        super().__init__(range_url)

    def get_headers(self):
        """Gets aom requests headers"""
        headers = {}
        self._aom_auth.set_auth_info(headers)

        return headers


def create_aom_auth(auth_type: str, auth_info: dict) -> AomAuth:
    """Creates AomAdapter auth base on auth type and auth info"""
    if auth_type == 'appcode':
        return AppCodeAuth(auth_info.get('app_code'))
    elif auth_type == 'token':
        return TokenAuth(auth_info)
    raise AuthenticationError('Unsupported aom auth type: {}, please check'.format(auth_type))


def create_aom_collector(aom_conf: AomConfig) -> AomAdapter:
    """Creates AomAdapter collectors"""
    aom_auth = create_aom_auth(aom_conf.auth_type, aom_conf.auth_info)
    return AomAdapter(aom_conf.base_url, aom_conf.project_id, aom_auth)
