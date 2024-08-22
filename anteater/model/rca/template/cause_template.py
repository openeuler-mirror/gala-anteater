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


from datetime import datetime
from .template import Template


class CauseTemplate(Template):
    """The app anomaly template"""

    def __init__(self, timestamp, machine_id, metric, entity_name):
        """The app anomaly template initializer"""
        super().__init__(timestamp, machine_id, metric, entity_name)

    def get_template(self):
        """Gets the template for app level anomaly events"""
        timestamp = round(self.timestamp.timestamp() * 1000)

        result = {
            'Timestamp': timestamp,
            'event_id': f'{timestamp}_{self.entity_id}',
            'Attributes': {
                'event_id': f'{timestamp}_{self.entity_id}',
            },
            'Resource': {
                'abnormal_kpi'
                'cause_metrics': self.cause_metrics
            },
            'desc': ""
        }
        for i in range(len(self.cause_metrics)):
            result[f'top{i}'] = self.cause_metrics[i]

        result['keywords']
        result['SeverityText'] = 'WARN',
        result['SeverityNumber'] = 13,
        result['Body'] = 'A cause inferring event for an abnormal event'

        if "@" in result["Attributes"]["cause_metric"]["metric"]:
            result["Attributes"]["cause_metric"]["metric"] = result["Attributes"]["cause_metric"]["metric"].split("@")[
                0]

        if "@" in result["Resource"]["metric"]:
            result["Resource"]["metric"] = result["Resource"]["metric"].split("@")[0]

        for m in result["Resource"]["cause_metrics"]:
            if "@" in m["metric"]:
                m["metric"] = m["metric"].split("@")[0]

        date_array = datetime.utcfromtimestamp(int(self.timestamp.timestamp()))
        # with open(os.path.join("/home/zhaoyongxin/usad_test/rcl_test_case", result['Attributes']['entity_id'].split('_')[0], f"CauseJson_{date_array.strftime('%Y-%m-%d-%H-%M-%S')}.json"), "w") as f:
        #     json.dump(result, f, indent=4, ensure_ascii=False)
        #     f.close()

        return result
