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
    json_filename = (
        get_input_data_path() + 'appdetails/' + 'appID_' + str(app_id) + '.json'
    )

    with open(json_filename, 'r', encoding='utf8') as f:
        app_details = json.load(f)

    return app_details


def get_image_file_name(app_id, sub_folder_name=None):
    if sub_folder_name is None:
        fixed_subfolder_name = ''
    else:
        if sub_folder_name.endswith('/'):
            fixed_subfolder_name = sub_folder_name
        else:
            fixed_subfolder_name = sub_folder_name + '/'

    full_folder_path = get_output_data_path() + fixed_subfolder_name
    Path(full_folder_path).mkdir(parents=True, exist_ok=True)

    image_file_name = full_folder_path + app_id + '.jpg'

    return image_file_name


def get_banner_file_name(app_id):
    banner_file_name = get_image_file_name(app_id, sub_folder_name=None)

    return banner_file_name


def get_screenshot_file_name(app_id, screenshot_index=0, is_thumbnail=True):
    sub_folder_name = 'screenshot_' + str(screenshot_index)

    if is_thumbnail:
        sub_folder_name += '_thumbnail'

    screenshot_file_name = get_image_file_name(app_id, sub_folder_name=sub_folder_name)

    return screenshot_file_name


def get_background_file_name(app_id):
    background_file_name = get_image_file_name(app_id, sub_folder_name='background')

    return background_file_name


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


async def main(image_type='banner', screenshot_index=0, is_thumbnail=True):
    async with aiohttp.ClientSession() as session:
        for app_id in sorted(get_app_ids(), key=int):
            if image_type == 'screenshot':
                banner_file_name_as_str = get_screenshot_file_name(
                    app_id,
                    screenshot_index,
                    is_thumbnail,
                )
            elif image_type == 'background':
                banner_file_name_as_str = get_background_file_name(app_id)
            else:
                banner_file_name_as_str = get_banner_file_name(app_id)

            banner_file_name = Path(banner_file_name_as_str)

            if banner_file_name.exists():
                continue

            app_details = get_app_details(app_id)

            try:
                app_type = app_details['type']
            except (KeyError, TypeError):
                continue

            if app_type == 'game':
                try:
                    if image_type == 'screenshot':
                        banner_url = get_screenshot_url(
                            app_details,
                            screenshot_index,
                            is_thumbnail,
                        )
                    elif image_type == 'background':
                        banner_url = get_background_url(app_details)
                    else:
                        banner_url = get_banner_url(app_details)
                except KeyError:
                    continue

                # Reference: https://stackoverflow.com/a/51745925
                async with session.get(banner_url) as resp:
                    if resp.status == 200:
                        f = await aiofiles.open(banner_file_name, mode='wb')
                        await f.write(await resp.read())
                        await f.close()
                        print(
                            'Banner downloaded to {} for appID {}.'.format(
                                banner_file_name,
                                app_id,
                            ),
                        )
                    else:
                        print(
                            'Banner for appID {} could not be downloaded.'.format(
                                app_id,
                            ),
                        )

    return


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
