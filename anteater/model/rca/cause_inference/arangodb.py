from pyArango.connection import Connection
from pyArango.database import Database
from pyArango.theExceptions import AQLQueryError

from anteater.utils.log import logger
_TIMESTAMP_COLL_NAME = 'Timestamps'
_OBSERVE_ENTITY_COLL_PREFIX = 'ObserveEntities'

CODE_OF_EDGE_COLL_NOT_FOUND = 404


def _get_collection_name(collection_type, ts_sec):
    return '{}_{}'.format(collection_type, ts_sec)


def connect_to_arangodb(arango_url, db_name, user_name="root", password=""):
    try:
        conn: Connection = Connection(arangoURL=arango_url)
        # conn: Connection = Connection(arangoURL=arango_url, username=user_name, password=password)
    except ConnectionError as ex:
        raise Exception('Connect to arangodb error because {}'.format(ex)) from ex
    if not conn.hasDatabase(db_name):
        raise Exception('Arango database {} not found, please check!'.format(db_name))
    return conn.databases[db_name]


def query_all(db, aql_query, bind_vars=None, raw_results=True):
    res = []
    query_hdl = db.AQLQuery(aql_query, bindVars=bind_vars, rawResults=raw_results)
    for item in query_hdl:
        res.append(item)
    return res


def query_recent_topo_ts(db: Database, ts) -> int:
    bind_vars = {'@collection': _TIMESTAMP_COLL_NAME, 'ts': ts}
    aql_query = '''
    FOR t IN @@collection
      FILTER TO_NUMBER(t._key) <= @ts
      SORT t._key DESC
      LIMIT 1
      RETURN t._key
    '''
    try:
        query_res = query_all(db, aql_query, bind_vars)
        logger.info(f"{query_res}")
    except AQLQueryError as ex:
        raise DBException(ex) from ex
    if len(query_res) == 0:
        raise DBException('Can not find topological graph at the abnormal timestamp {}'.format(ts))
    last_ts = query_res[0]
    return int(last_ts)


def gen_filter_str(query_options) -> str:
    if not query_options:
        return ''

    filter_options = ['v.{} == @{}'.format(k, k) for k in query_options]
    filter_str = 'filter ' + ' and '.join(filter_options)
    return filter_str




