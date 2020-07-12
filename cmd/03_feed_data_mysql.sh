source .env

$docker cp \
    $ROOT_DIR/sql/01_create_table \
        $MYSQL_CONTAINER_NAME:$MYSQL_FILE_PATH/01_create_table

$docker exec \
    -i \
    $MYSQL_CONTAINER_NAME \
    mysql -u$MYSQL_USER -p$MYSQL_USER_PASSWORD \
    < ./sql/01_create_table

$docker cp \
    $ROOT_DIR/sql/02_insert_data \
        $MYSQL_CONTAINER_NAME:$MYSQL_FILE_PATH/02_insert_data

$docker cp \
    $ROOT_DIR/data/raw/bs140513_032310.csv \
        $MYSQL_CONTAINER_NAME:$MYSQL_FILE_PATH/bs140513_032310.csv

$docker exec \
    -i \
    $MYSQL_CONTAINER_NAME \
    mysql -u$MYSQL_USER -p$MYSQL_USER_PASSWORD \
    <<< "$(envsubst < ./sql/02_insert_data)"
