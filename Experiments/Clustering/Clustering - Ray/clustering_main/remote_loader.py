import os
import sys
import importlib
from time import sleep
import tempfile
from glob import glob

import ray


def remote_import(paths, wait=10):
    """Dynamically copy and import local custom modules and packages to all Ray workers

    Call this function after starting Ray up but before executing any remote operation (task or actor) that depends
     on the custom modules.

    The modules/packages are copied and imported every time the function is called,
     even if the local files have changed. Only .py files will be copied, any other extension will be ignored.

    The actual files will exist on a secure temporary folder on the remote machine just for the time needed to
    import them to the global namespace, then they are deleted.

    After the function returns, you may run any kind of import statement on the remote workers
    as if the packages were installed there. Please note that this is obtained with a rather dirty hack,
    polluting the global namespace. Be sure that none of the custom package names collides with other packages.

    :param paths: iterable of all locations you want to scan for python packages and modules.
    :param wait: number of seconds to wait for all workers to complete the module copying and loading.
    Since the underlying Ray method run_function_on_all_workers does not allow you to explicitly wait for
    these tasks to complete. Defaults to 10.
    """
    
    copy_to_remote = {}

    for module_path in paths:  # /look/here/for/packages
        for filename in glob(os.path.join(module_path, '**', '*.py'), recursive=True):
            with open(filename) as f:
                # /look/here/for/packages/my_package/my_module.py ==> my_package/my_module.py
                subpath = filename.replace(module_path, '').strip('/')
                copy_to_remote[subpath] = ray.put(f.read())  # the file content is sent to Ray object store

    # This function will run on every worker
    def on_remote(_):

        modules_to_be_imported = set()

        with tempfile.TemporaryDirectory(prefix='ray_custom_remote_modules') as temp_folder:
            # Retrieve files from Ray object store and copies them into a local tempfolder (including subfolders).
            for subpath, content_id in copy_to_remote.items():

                subpath_chunks = subpath.replace('.py', '').split(os.sep)  # ['package', 'subpackage', 'module']
                for i in range(len(subpath_chunks)):
                    # ['package', 'package.subpackage',      'package.subpackage.module'] will be added to global namespace
                    modules_to_be_imported.add('.'.join(subpath_chunks[:i+1]))

                target_path = os.path.join(temp_folder, subpath)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, 'w+') as f:
                    f.write(ray.get(content_id))

            sys.path.append(temp_folder)  # Add tempfolder to the sys path
            importlib.invalidate_caches()  # Clear import caches so that the new folder can be seen and read

            for module_name in modules_to_be_imported:
                module = importlib.import_module(module_name)  # Programmatically import the module
                module = importlib.reload(module)  # (in case files have been modified, this will re-read them)
                print('Importing', module_name)
                globals()[module_name] = module  # available to global namespace

            # tempfolder is deleted

    ray.worker.global_worker.run_function_on_all_workers(on_remote)
    sleep(wait)
