source .env

$docker pull mysql/mysql-server:latest
# TODO: get auto-generated password from docker logs

mkdir -p $(MYSQL_MOUNT_LOC)

$docker exec \
    -i \
    $MYSQL_CONTAINER_NAME \
    mysql -uroot -p$MYSQL_ROOT_PASSWORD \
    <<< "CREATE DATABASE $MYSQL_DB_NAME;
         SHOW DATABASES;"

$docker exec \
    -i \
    $MYSQL_CONTAINER_NAME \
    mysql -uroot -p$MYSQL_ROOT_PASSWORD \
    <<< "USE $MYSQL_DB_NAME;
         CREATE USER '$MYSQL_USER'@'localhost'
         IDENTIFIED BY '1';
         GRANT ALL ON $MYSQL_DB_NAME.* TO '$MYSQL_USER'@'localhost';"
