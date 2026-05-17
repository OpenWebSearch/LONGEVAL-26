# docker build -t webis/longeval-pyterrier-baseline:dev-0.0.2 .
FROM webis/longeval-pyterrier-baseline:dev-0.0.1

RUN pip3 uninstall -y ir-datasets-longeval \
	&& pip3 install ir-datasets-longeval
