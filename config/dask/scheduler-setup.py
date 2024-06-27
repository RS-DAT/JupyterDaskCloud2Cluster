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

    Note, however, that the workers will still advertise their
    "real" address, which is likely on the private network of the
    SLURM cluster (10.X.X.X). For the scheduler to reach them, we
    thus need to redirect all traffic towards these IP addresses to
    the localhost, e.g. with the following iptables rule (replace
    XXXX:YYYY with the range of ports used by the workers, e.g.
    8900:9900 - see `jobqueue.yaml` for the range employed):

    ```
    sudo iptables -t nat -I OUTPUT --dst 10.0.0.0/16 -p tcp --match multiport --dports XXXX:YYYY -j REDIRECT
    ```

    Also note that it is assumed that the workers can reach the
    scheduler: the port which the scheduler binds to should be open
    on the machine where the scheduler runs on.
    """
    def __init__(self, slurm_user=None, slurm_host=None, slurm_ssh_key_filename=None):

        slurm_user = os.environ.get('SLURM_USER', None) \
            if slurm_user is None else slurm_user
        slurm_host = os.environ.get('SLURM_HOST', None) \
            if slurm_host is None else slurm_host

        # SSH key is not required, we can be using a SSH agent
        if (slurm_user is None) or (slurm_host is None):
            raise ValueError('Provide SLURM user and host names')

        slurm_ssh_key_filename = os.environ.get('SLURM_SSH_KEY_FILENAME', None) \
            if slurm_ssh_key_filename is None else slurm_ssh_key_filename
        connect_kwargs = {"key_filename": slurm_ssh_key_filename} \
            if slurm_ssh_key_filename is not None else None

        self.forward_locals = {}
        self.connection = fabric.Connection(
            user=slurm_user, host=slurm_host, connect_kwargs=connect_kwargs,
        )
        super().__init__()

    def add_worker(self, scheduler, worker):
        """
        When a worker starts, local forwarding of the worker's port.
        """
        host, port = get_address_host_port(worker)
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
@click.option("--slurm-ssh-key-filename", type=str)
def dask_setup(scheduler, slurm_user=None, slurm_host=None, slurm_ssh_key_filename=None):
    plugin = SchedulerPluginForRemoteWorkers(slurm_user, slurm_host, slurm_ssh_key_filename)
    scheduler.add_plugin(plugin)

