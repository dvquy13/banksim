source .env

$docker exec \
    -i \
    $MYSQL_CONTAINER_NAME \
    mysql -u$MYSQL_USER -p$MYSQL_USER_PASSWORD \
    < $1
