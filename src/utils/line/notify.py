import json
from textwrap import wrap
from requests import post


def line_notify(
    msg: str,
    width: int=10000,
    url = 'https://notify-api.line.me/api/notify'
) -> None:

    with open('./src/utils/line/line.json') as f:
        d = json.loads(f.read())
    headers = {'Authorization': f'Bearer {d["token"]}'}

    for m in wrap(msg, width=width):
        data = {'message': m}
        post(url=url, headers=headers, data=data)
