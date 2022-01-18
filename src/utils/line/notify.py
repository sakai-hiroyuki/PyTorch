import json
from textwrap import wrap
from requests import post


def line_notify(
    msg: str,
    token: str,
    width: int=10000,
    url: str='https://notify-api.line.me/api/notify'
) -> None:
    '''
    Line Notify(https://notify-bot.line.me/ja/)へ通知を送る.
    
    Parameters
    ----------
    msg: str
        送るメッセージ.

    token: str
        Line Notifyのトークン.
    
    width: int=10000
        Lineで送信可能な最大文字数.
    
    url: str='https://notify-api.line.me/api/notify'
        Line NotifyのAPIのURL.
    '''

    headers = {'Authorization': f'Bearer {token}'}

    for m in wrap(msg, width=width):
        data = {'message': m}
        post(url=url, headers=headers, data=data)
