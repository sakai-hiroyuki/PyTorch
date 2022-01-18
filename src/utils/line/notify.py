import json
from textwrap import wrap
from requests import post


def line_notify(
    msg: str,
    token: str,
    width: int=10000,
    url = 'https://notify-api.line.me/api/notify'
) -> None:
    '''
        https://notify-bot.line.me/ja/
    '''

    headers = {'Authorization': f'Bearer {token}'}

    for m in wrap(msg, width=width):
        data = {'message': m}
        post(url=url, headers=headers, data=data)
