.PHONY: all

docker_fn=$(docker)

# MySQL
mysql-run:
	$(docker_fn) run \
		--name=$(MYSQL_CONTAINER_NAME) \
		-t \
		-d \
		--env="MYSQL_ROOT_PASSWORD=$(MYSQL_ROOT_PASSWORD)" \
		--publish 6603:3306 \
		--volume=$(MYSQL_MOUNT_LOC):/var/lib/mysql \
		mysql/mysql-server:latest

mysql-start:
	$(docker_fn) start $(MYSQL_CONTAINER_NAME)

mysql-stop:
	$(docker_fn) stop $(MYSQL_CONTAINER_NAME)

mysql-bash:
	@$(docker_fn) exec -it $(MYSQL_CONTAINER_NAME) mysql -uroot -p$(MYSQL_ROOT_PASSWORD)

# Conda
install-conda-env:
	conda env create -f conda.yaml

update-conda-env:
	conda env update banksim --file conda.yaml --prune
