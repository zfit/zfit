#  Copyright (c) 2024 zfit

from __future__ import annotations

import contextlib
import functools
import multiprocessing
import os
import sys
import typing

import tensorflow as tf
from dotmap import DotMap

from .deprecation import deprecated
from .exception import BreakingAPIChangeError, IllegalInGraphModeError
from .temporary import TemporarilySet


class RunManager:
    DEFAULT_MODE: typing.ClassVar = {"graph": "auto", "autograd": True}

    def __init__(self, n_cpu="auto"):
        """Handle the resources and runtime specific options.

        The `run` method is equivalent to `sess.run`
        """
        self.MAX_CHUNK_SIZE = sys.maxsize
        self.chunking = DotMap()
        self._cpu = []
        self._n_cpu = None
        self._inter_cpus = None
        self._intra_cpus = None
        self._strict = False
        self.numeric_checks = True
        self._mode = self.DEFAULT_MODE.copy()
        self.set_n_cpu(n_cpu=n_cpu)
        self._hashing_enabled = True

        # TODO: keep this?
        self._enable_parameter_autoconversion = True

        # set default values
        self.chunking.active = False  # not yet implemented the chunking...
        self.chunking.max_n_points = 1000000

    @property
    def mode(self):
        return self._mode

    @property
    def chunksize(self):
        if self.chunking.active:
            return self.chunking.max_n_points
        else:
            return self.MAX_CHUNK_SIZE

    @property
    def n_cpu(self):
        return len(self._cpu)

    def set_n_cpu(self, n_cpu: str | int = "auto", strict: bool = False) -> None:
        """Set the number of cpus to be used by zfit. For more control, use `set_cpus_explicit`.

        Args:
            n_cpu: Number of cpus, will be the number for inter-op parallelism
            strict: If strict, sets intra parallelism to 1
        """
        if n_cpu == "auto":
            try:
                cpu = sorted(os.sched_getaffinity(0))
            except AttributeError:
                cpu = range(multiprocessing.cpu_count())
        elif isinstance(n_cpu, int):
            cpu = range(n_cpu)
        self._cpu = [f"dummy_cpu{i}" for i in cpu]
        n_cpu = len(cpu)
        if strict ^ self._strict:
            intra = 1 if strict else 2
            inter = n_cpu
            self.set_cpus_explicit(intra=intra, inter=inter)

    def set_cpus_explicit(self, intra: int, inter: int) -> None:
        """Set the number of threads (cpus) used for inter-op and intra-op parallelism.

        Args:
            intra: Number of threads used to perform an operation. For larger operations, e.g. large Tensors, this
                is usually beneficial to have >= 2.
            inter: Parallelization on the level of ops. This is beneficial, if many operations can be computed
                independently in parallel.
        """
        try:
            tf.config.threading.set_intra_op_parallelism_threads(intra)
            tf.config.threading.set_inter_op_parallelism_threads(inter)
            self._n_cpu = inter + intra
        except RuntimeError as err:
            msg = (
                "Cannot set the number of cpus after initialization, has to be at the beginning."
                f" Original message: {err}"
            )
            raise RuntimeError(msg) from err

    @contextlib.contextmanager
    def aquire_cpu(self, max_cpu: int = -1) -> list[str]:
        if isinstance(max_cpu, int):
            if max_cpu < 0:
                max_cpu = max((self.n_cpu + 1 + max_cpu, 0))  # -1 means all
            if max_cpu == 0:
                cpu = []
            else:
                n_cpu = min((max_cpu, self.n_cpu))

                cpu = self._cpu[-n_cpu:]
                self._cpu = self._cpu[:-n_cpu]

            yield cpu
            self._cpu.extend(cpu)

    @deprecated(
        None,
        "The `run(...)` method originates from the old `tf.Session` API in TF1 and is not needed anymore. In most cases, it can simply be removed. "
        "It's current functionality converts the input to numpy, which most of the time is not needed anyways."
        "Alternatively, use `znp.asarray(...)` to convert to zfit-friendly array (that acts nearly like numpy) or `np.asarray(...) if a numpy"
        " array is *really needed*.",
        "Remove the `run(...)` method.",
    )
    def __call__(self, *args, **kwargs):
        # TODO: catch maybe sets, as they change the number of elements if we have identical ones
        # and convert them. Before it's fine, e.g. Parameters are unique, but after it's a value.
        if kwargs:
            msg = "Why kwargs provided?"
            raise RuntimeError(msg)

        flattened_args = tf.nest.flatten(args)
        evaluated_args = [eval_object(arg) for arg in flattened_args]
        values = tf.nest.pack_sequence_as(args, flat_sequence=evaluated_args)

        if len(args) == 1:
            values = values[0]
        return values

    @staticmethod
    @deprecated(date=None, instructions="Use `set_graph_mode(False)`")
    def experimental_enable_eager(eager: bool = False):  # noqa: ARG004
        """DEPRECEATED! Enable eager makes tensorflow run like numpy. Useful for debugging.

        Do NOT directly mix it with Numpy (and if, also enable the numberical gradient).

        This can BREAK in the future.
        """
        msg = "This function is deprecated. Use `set_graph_mode(False)` instead."
        raise BreakingAPIChangeError(msg)

    def experimental_disable_param_update(self, value: bool = True):
        """Disable the automatic update of parameters in the minimization temporarily or permanently.

        Can be used with a context manager (recommended!).

        This can have unintended side effects if other code (i.e. hepstats) is used, as they expect the parameters to be
        updated usually.

        Instead, `result.update_params()` has to be called manually to update the parameters if needed.
        """

        from zfit import settings

        def setter(v):
            settings.options.auto_update_params = not v

        return TemporarilySet(value=value, setter=setter, getter=lambda: not settings.options.auto_update_params)

    def set_graph_mode(self, graph: bool | str | dict | None = None, *, set_backend: bool | None = None):
        """Set the policy for graph building and the usage of automatic vs numerical gradients.

        zfit runs on top of TensorFlow, a modern, powerful computing engine very similar in design to Numpy.
        An interactive tutorial can be found at https://github.com/zfit/zfit-tutorials

        **Graph building**

        It has two ways to be run where the first
        defaults to the normal mode we are in except inside a :py:func:`~zfit.z.function` decorated function. Setting
        the mode allows to control the behavior of decorated functions to not always trigger a graph building.

         - **numpy-like/eager**: in this mode, the syntax slightly differs from pure numpy but is similar. For example,
            `tf.sqrt`, `tf.math.log` etc. The return values are `EagerTensors` that represent "wrapped Numpy arrays" and
            can directly be used with any Numpy function. They can explicitly be converted to a Numpy array with
            `znp.asarray(EagerTensor)`, which takes also care of nested structures and already existing `np.ndarrays`,
            or just a `.numpy()` method.
            The difference to Numpy is that TensorFlow tries to optimize the calculation slightly beforehand and may
            also executes on the GPU. This will result in a slight performance penalty for *very small* computations
            compared to Numpy, on the other hand an improved performance for larger computations.

         - **graph**: a function can be decorated with :py:func:`~zfit.z.function`, which will *not* execute its
            content immediately, but first trace it and build a graph. This is done by *recording all `tf.*` operations
            and adding them to the graph while any Python operation, e.g. `np.random.*` will use a fixed value added
            to the graph. Building a graph greatly reduces the flexibility, since only `tf.*` operations can
            effectively be used to have dynamics in there, on the other hand it can greatly increase the performance.
            When the graph is built, it is cached (for later re-use), optimized and then executed.
            Calling a `tf.function` decorated
            function does therefore not make an actualy difference *for the caller*. But it is a difference on how the
            function behaves.

            .. code-block:: python

                @z.function
                def add_rnd(x):
                     res1 = x + np.random.uniform()  # returns a Python scalar. This exact scalar will be constant
                     res2 = x + z.random.uniform(shape=())  # returns a tf.Tensor. This will be flexible
                     return res1, res2

                res_np1, res_tf1 = add_rnd(5)
                res_np2, res_tf2 = add_rnd(5)

                assert res_np1 == res_np2  # they will be the same!
                assert res_tf1 != res_tf2  # these differ

            While writing TensorFlow is just like Numpy, if we build a graph, only `tf.*` dynamics "survives".
            Important: while values are usually constant, changing a :py:class:`zfit.Parameter` value with
            :py:meth:`~zfit.Parameter.set_value(...)` *will* change the value in the graph as well.

                        .. code-block:: python

                @z.function
                def add(x, param):
                     return x + param

                param = zfit.Parameter('param1', 36)
                assert add_rnd(5, param) == 41
                param.set_value(6)
                assert add_rnd(5, param) == 42  # the value changed!

            Every graph generation takes some additional time and is stored, consuming memory and slowing down the
            overall execution process.
            To clear all caches and force a rebuild of the graph, `zfit.run.clear_graph_cache()` can be used.

            If a function is not decorated with `z.function`, this does not guarantee that it is executed in eager,
            as an outer function may uses a decorator. A typical case is the loss, which is decorated. Therefore,
            any Model called inside will be evaluated with a graph building first.


            **When to use what**:
             - Any repeated call (as a typical call to the loss function in the minimization process) is usually
               better suited within a `z.function`.
             - A single call (e.g. for plotting) or repeated calls *with different arguments* should rather be run
               *without* a graph built first
             - Debugging is usually way easier without graph building. Therefore, set the graph mode to `False`
             - If the minimization fails but the pdf works without graph, maybe the graph mode can be switched on
               for everything to have the same behavior in the pdf as when the loss is called.

        Args:
            graph: Policy for when to build a graph with which function. Currently allowed values are
              - `True`: this will make all :py:func:`zfit.z.function` decorated function to be traced. Useful
                to have a consistent behavior overall, as e.g. a PDF may not be traced if `pdf` or `integrate` is
                called, but may be traced when inside a loss.
              - `False`: this will make everything execute immediately, like Numpy (this is **not enough** to be
                fully Numpy compatible in the sense of using `, also see the `autograd` option)
              - 'auto': Something in between, where sampling (currently) and the loss builds a graph but
                all model methods, such as `pdf`, `integrate` (except of `*sample*`) do not and are executed
                eagerly.
              - (**advanced and experimental!**): a dictionary containing the string of a wrapped function identifier
                (see also :py:func:`~zfit.z.function` for more information about this) with a boolean that switches
                explicitly on/off the graph building for this type of decorated functions.
            set_backend: if true, also set the mode for the backend (if mode is set to `False`). This is usually desired, as
                the backend (TensorFlow) can have compiled functions as well, especially in control flows that are used for example
                in binned PDFs. This can also be manually achieved using ``tf.config.run_functions_eagerly``. Setting this to false
                *can* have deadlocks with threads.
        """
        if not tf.executing_eagerly():
            msg = (
                "Cannot change the execution mode of graph inside a `z.function`"
                " decorated function. Only possible in an eager context."
            )
            raise IllegalInGraphModeError(msg)
        if set_backend is None:
            set_backend = True
        return self._force_set_graph_mode(graph, set_backend=set_backend)

    def _force_set_graph_mode(self, graph, set_backend=True):
        if graph is None:
            graph = "auto"
        mode_setter = functools.partial(self._set_graph_mode, set_backend=set_backend)
        return TemporarilySet(value=graph, setter=mode_setter, getter=self.get_graph_mode)

    def set_autograd_mode(self, autograd: bool | None = None):
        """Use automatic or numerical gradients.

        zfit runs on top of TensorFlow, a modern, powerful computing engine very similar in design to Numpy.
        An interactive tutorial can be found at https://github.com/zfit/zfit-tutorials

        **automatic gradient**

        A strong feature of TensorFlow is the possibility to derive an analytic expression for the gradient
        by successively applying the chain rule to all of its operations. This is *independent* of whether the code
        is run in graph or eager execution, but requires all operations that are dynamic to be `tf.*` operations.
        For example, multiplying by a constant (constant as in *not chaning ever*) does not require the constant to
        be a `tf.constant(...)` but can be a Python scalar. For example, it is also fine to use a fixed template shape
        using Numpy (Scipy etc), as the template shape will stay constant (this requires though to use a
        `z.numpy_function` to work, but this is another story about graph mode or not).

        To allow to have dynamic numpy operations in a component, preferably wrapped with `z.numpy_function` instead of
        forced eager, and to still retrieve a meaningful gradient, a numerical gradient has to be used.
        In general, this can be achieved by setting the `autograd` to False. Any derivative received will then be
        numerically computed. Furthermore, some minimizers (e.g. :py:class:`~zfit.minimize.Minuit`) have their own way
        of calculating gradients, which can be faster.
        Disabling `autograd` and using the zfit builting numerical way of calculating the gradients and hessian can
        be less stable and may raises errors.

        Args:
            autograd: Whether the automatic gradient feature of TensorFlow should be used or a numerical procedure
              instead. If any non-constant Python (numpy, scipy,...) code is used inside, this should be switched on.
        """
        if autograd is None:
            autograd = True
        return TemporarilySet(
            value=autograd,
            setter=self._set_autograd_mode,
            getter=self.get_autograd_mode,
        )

    def set_mode(
        self,
        graph: bool | str | dict | None = None,  # noqa: ARG002
        autograd: bool | None = None,  # noqa: ARG002
    ):
        """DEPRECATED!

        Use `set_graph_mode` or `set_autograd_mode`.
        """
        msg = "Use `set_graph_mode` or `set_autograd_mode`."
        raise BreakingAPIChangeError(msg)

    def _set_autograd_mode(self, autograd):
        if autograd is not None:
            from zfit import settings

            settings.options.numerical_grad = not autograd
            self._mode["autograd"] = autograd

    def _set_graph_mode(self, graph, *, set_backend=True):
        if graph is None:
            graph = "auto"
        from .graph import jit as jit_obj

        # only run eagerly if no graph
        if set_backend:
            tf.config.run_functions_eagerly(graph is False)
        if graph is True:
            jit_obj._set_all(True)
        elif graph is False:
            jit_obj._set_all(False)
        elif graph == "auto":
            jit_obj._set_default()
        elif isinstance(graph, dict):
            jit_obj._update_allowed(graph)
        elif graph is not None:
            msg = (
                f"{graph} is not a valid keyword to the `jit` behavior. Use either "
                f"True, False, 'default' or a dict. You can read more about it in the docs."
            )
            raise ValueError(msg)
        if graph is not None:
            self._mode["graph"] = graph

    def get_graph_mode(self) -> bool | str:
        """Return the current policy for graph building.

        Returns:
            The current policy. For more information, check :py:meth:`~zfit.run.set_mode`.
        """
        return self.mode["graph"]

    @deprecated(None, "Use `get_graph_mode` instead.")
    def current_policy_graph(self) -> bool | str:
        """DEPRECEATED!

        Use `get_graph_mode` instead.
        """
        msg = "Use `get_graph_mode` instead."
        raise BreakingAPIChangeError(msg)

    def get_autograd_mode(self) -> bool:
        """The current policy for using the automatic gradient or falling back to the numerical.

        Returns:
            If autograd is being used.
        """
        return self.mode["autograd"]

    def current_policy_autograd(self) -> bool:
        """DEPRECATED!

        Use `get_autograd_mode` instead.
        """
        msg = "Use `get_autograd_mode` instead."
        raise BreakingAPIChangeError(msg)

    def set_mode_default(self):
        """Reset the mode to the default of `graph` = 'auto' and `autograd` = True."""
        self.set_autograd_mode(True)
        self.set_graph_mode("auto")

    def clear_graph_cache(self, *, call_gc: bool | None = None):
        """Clear all generated graphs and effectively reset. Should not affect execution, only performance.

        In a simple fit scenario, this is not used. But if several fits are performed with different python objects such
        as a scan over a range (by changing the norm_range and creating a new dataset), doing minimization and therefore
        invoking the loss (by default creating a graph) will leave the graphs in the cache, even tough the already
        scanned ranges are not needed anymore.

        To clean, this function can be invoked. The only effect should be to speed up things, but should not have any
        side-effects other than that.

        Args:
            call_gc: If `True`, call the garbage collector after clearing the cache. This can help if there are issues with non-cleared graphs.
           (it can impact the performance if called in a loop, as the gc can take up to a second)
        """
        if call_gc is None:
            call_gc = False
        from zfit.util.cache import clear_graph_cache

        clear_graph_cache(call_gc=call_gc)

    def set_graph_cache_size(self, size: int | None = None):
        """Set the size of the graph cache to the same value for all.

        Whenever a function, decorated with `z.function` is called, it is first compiled to a graph, which is cached.
        For different reasons, there can be different compiled functions of the same Python function (such as changed
        internal parameters). The cache determines how many compiled functions are kept in memory.

        Args:
            size:(default=10) The size of the cache. If None, the default size is used. With a lower number, a
                smaller memory footprint *can* be achieved in some cases, but the runtime *can* be slower in some cases
                (they do not need to be the same). Potentially, the cache should be at least of the size as the number
                of calls to a function *with different arguments* is expected to happen *outside of any loop*/within
                one execution of a loop.
        """
        from zfit.z.zextension import FunctionWrapperRegistry

        if size is not None and size < 1:
            msg = "The size of the cache must be at least 1."
            raise ValueError(msg)

        for registry in FunctionWrapperRegistry.registries:
            registry.set_graph_cache_size(size)

    def assert_executing_eagerly(self):
        """Assert that the execution is eager and Python side effects are taken into account.

        This can be placed inside a model *in case python side-effects are necessary* and no other way is possible.
        """
        if not tf.executing_eagerly():
            msg = "This code is not supposed to run inside a graph."
            raise RuntimeError(msg)

    @property
    @deprecated(None, "Use `executing_eagerly` instead.")
    def experimental_is_eager(self):
        msg = "Use `executing_eagerly` instead."
        raise BreakingAPIChangeError(msg)

    def executing_eagerly(self):
        """Whether eager execution is enabled or not.

        If inside a `z.function` decorated function and it is being compiled, this will return `False`.

        Returns:
            True if eager execution is enabled, False if not.
        """
        return tf.executing_eagerly()

    @deprecated(date=None, instructions="Use `clear_graph_cache` instead.")
    def experimental_clear_caches(self):
        """DEPRECATED!

        Use `clear_graph_caches` instead.
        """
        msg = "Use `clear_graph_cache` instead."
        raise BreakingAPIChangeError(msg)

    def hashing_data(self):
        """If hashing of data (required for caching) is enabled."""
        return self._hashing_enabled

    def set_data_hashing(self, enabled: bool):
        """Enable or disable hashing of data (required for caching).

        Args:
            enabled: Whether hashing of data is enabled.
        """
        self._hashing_enabled = enabled


@deprecated(None, "Use np.array(obj) instead.")
def eval_object(obj: object) -> object:
    from zfit.core.parameter import BaseComposedParameter

    if isinstance(obj, BaseComposedParameter):  # currently no numpy attribute. Should we add this?
        obj = obj.value()
    if tf.is_tensor(obj):
        return obj.numpy()
    else:
        return obj
