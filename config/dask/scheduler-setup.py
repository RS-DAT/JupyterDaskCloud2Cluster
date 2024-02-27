"""
For examples on Dask plugins see docs:

* https://distributed.dask.org/en/latest/plugins.html
* https://docs.dask.org/en/latest/customize-initialization.html

"""
import logging
import os

import click
import fabric

from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.comm import get_address_host_port


logger = logging.getLogger(__name__)


class SchedulerPluginForRemoteWorkers(SchedulerPlugin):
    """
    Scheduler plugin to connect to workers running on a remote
    SLURM system. When a new worker is added to the cluster, 
    the port to which the worker binds is forwarded locally.
    """
    def __init__(self, slurm_user=None, slurm_host=None):

        slurm_user = os.environ.get('SLURM_USER', None) if slurm_user is None else slurm_user
        slurm_host = os.environ.get('SLURM_HOST', None) if slurm_host is None else slurm_host
        if (slurm_user is None) or (slurm_host is None):
            raise ValueError('Provide SLURM user and host names')

        self.forward_locals = {}
        self.connection = fabric.Connection(user=slurm_user, host=slurm_host)
        super().__init__()

    def add_worker(self, scheduler, worker):
        """
        When a worker starts, local forwarding of the worker's port.
        """
        nanny = scheduler.workers[worker].nanny
        host, _ = get_address_host_port(nanny)
        _, port = get_address_host_port(worker)
        forward_local = self.connection.forward_local(
            remote_port=port,
            local_port=port,
            remote_host=host,
        )
        # Connection.forward_local is a contextmanager, but we need
        # to keep forwarding alive for all the life of the worker
        forward_local.__enter__()
        self.forward_locals[worker] = forward_local

    def remove_worker(self, scheduler, worker, *, stimulus_id, **kwargs):
        """
        When a worker stops, close the corresponding connection.
        """
        forward_local = self.forward_locals.pop(worker, None)
        if forward_local is not None:
            # stop local forwarding
            forward_local.__exit__(None, None, None)
            logger.info(f'ended port forwarding for worker: {worker}')


@click.command()
@click.option("--slurm-user", type=str)
@click.option("--slurm-host", type=str)
def dask_setup(scheduler, slurm_user=None, slurm_host=None):
    plugin = SchedulerPluginForRemoteWorkers(slurm_user, slurm_host)
    scheduler.add_plugin(plugin)
