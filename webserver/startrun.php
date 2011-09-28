<?php 
	
  include 'settings.php';
  
	// If PHP's magic_quotes_gpc option is on, all special characters (e.g. quotes) in
	// GET/POST/COOKIE strings are automatically escaped. Here we undo this, because we want the
	// true input. I can do my own escaping if needed, thank you very much!
	if (get_magic_quotes_gpc())
	{
   		function traverse(&$arr)
   		{
       		if (!is_array($arr)) return;
       		foreach ($arr as $key => $val)
           		is_array($arr[$key]) ? traverse($arr[$key]) : ($arr[$key]=stripslashes($arr[$key]));
   		}
   		$gpc = array(&$_GET,&$_POST,&$_COOKIE);
   		traverse($gpc);
	} 
	
	$description = $_POST['description'];
	if (empty($description)) {
		//err('"description" argument is missing.');
		$description = '';
	}

	// Connect to MySQL server.
	$result = mysql_connect($host,$user,$password);
	if (!$result) err("Failed to connect to MySQL database!");

	// Select NVTB database.
	$result = mysql_select_db($database);
	if (!$result) err( "Unable to select database");

	// Check for job identifier
	$job = $_POST['job'];
	if (empty($job)) err('"job" argument is missing.');
	$job = "'".mysql_real_escape_string($job)."'";

	// Escape strings to be sent to database
	$description = mysql_real_escape_string($description);

	// Insert/replace the record.
	$sql = "INSERT INTO `runs` (`source`,`time`,`job`,`description`) VALUES ('$_SERVER[REMOTE_ADDR]',NOW(),$job,'$description')";
	$result = mysql_query($sql);
	if (!$result) err("Unable to insert or update row in table. SQL statement: $sql");
	
	// Get id of newly inserted record
	$id = mysql_insert_id();

	// Close MySQL connection.
	mysql_close();
	
	echo $id;
	flush();
	
	// No further output.
	//exit();
	
	function err($msg) {
		header("HTTP/1.1 500 Internal Server Error");
		echo $msg;
		exit();
	}

?>
