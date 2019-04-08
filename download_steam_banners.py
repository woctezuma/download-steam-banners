import asyncio
import json
from pathlib import Path

import aiofiles
import aiohttp


def get_user_name():
    user_name = 'Woctezuma'
    return user_name


def get_pycharm_path():
    pycharm_path = 'C:/Users/' + get_user_name() + '/PycharmProjects/'
    return pycharm_path


def get_input_data_path():
    data_path = get_pycharm_path() + 'steam-api/data/'

    return data_path


def get_output_data_path():
    data_path = get_pycharm_path() + 'download-steam-banners/data/'

    return data_path


def get_app_ids():
    successful_app_id_filename = get_input_data_path() + 'successful_appIDs.txt'

    with open(successful_app_id_filename, 'r') as f:
        parsed_app_ids = set([line.strip() for line in f])

    return parsed_app_ids


def get_app_details(app_id):
    json_filename = get_input_data_path() + 'appdetails/' + 'appID_' + str(app_id) + '.json'

    with open(json_filename, 'r', encoding='utf8') as f:
        app_details = json.load(f)

    return app_details


def get_banner_file_name(app_id):
    banner_file_name = get_output_data_path() + app_id + '.jpg'
    return banner_file_name


def get_banner_url(app_details):
    banner_url = app_details['header_image']

    return banner_url


def get_screenshot_url(app_details, screenshot_index=0, is_thumbnail=True):
    if is_thumbnail:
        screenshot_url = app_details['screenshots'][screenshot_index]['path_thumbnail']
    else:
        screenshot_url = app_details['screenshots'][screenshot_index]['path_full']

    return screenshot_url


def get_background_url(app_details):
    background_url = app_details['background']

    return background_url


async def main():
    async with aiohttp.ClientSession() as session:

        for app_id in sorted(get_app_ids(), key=int):
            banner_file_name = Path(get_banner_file_name(app_id))

            if banner_file_name.exists():
                continue

            app_details = get_app_details(app_id)

            try:
                app_type = app_details['type']
            except (KeyError, TypeError):
                continue

            if app_type == 'game':

                try:
                    banner_url = get_banner_url(app_details)
                except KeyError:
                    continue

                # Reference: https://stackoverflow.com/a/51745925
                async with session.get(banner_url) as resp:
                    if resp.status == 200:
                        f = await aiofiles.open(banner_file_name, mode='wb')
                        await f.write(await resp.read())
                        await f.close()
                        print('Banner downloaded to {} for appID {}.'.format(banner_file_name, app_id))
                    else:
                        print('Banner for appID {} could not be downloaded.'.format(app_id))

    return


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
