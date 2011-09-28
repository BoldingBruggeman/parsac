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

	// Obtain POST arguments
	$run   = $_POST['run'];
	$count = $_POST['count'];
	
	// Check presence of required arguments.
	if (empty($run)) err('Run argument is missing.');

	// Convert run identifier to integer.
	$run = (int)$run;
	
	// Connect to MySQL server.
	$result = mysql_connect($host,$user,$password);
	if (!$result) err("Failed to connect to MySQL database!");

	// Select database.
	$result = mysql_select_db($database);
	if (!$result) err( "Unable to select database");

	if (empty($count)) {
		// Only one result is being reported.
		$parameters   = $_POST['parameters'];
		$lnlikelihood = $_POST['lnlikelihood'];
		if (empty($parameters)) err('Parameter argument is missing.');
		insertrow($parameters,$lnlikelihood);
	}
	else {
		// A series of results is being reported.
		for ($i=0; $i<$count; $i++) {
			$parameters   = $_POST["parameters$i"];
			$lnlikelihood = $_POST["lnlikelihood$i"];
			if (empty($parameters)) err("Parameter argument for result $i is missing.");
			insertrow($parameters,$lnlikelihood);
		}
	}

	// Close MySQL connection.
	mysql_close();
	
	echo 'success!';
	flush();
	
	// No further output.
	//exit();
	
	function insertrow($parameters,$lnlikelihood) {
		global $run;
		
		// Escape strings to be sent to database
		$parameters = mysql_real_escape_string($parameters);
		
		// Handle the case where lnlikelihood == NULL
		if (strcmp($lnlikelihood,'')==0)
			$lnlikelihood = 'NULL';
		else
			$lnlikelihood = "'".mysql_real_escape_string($lnlikelihood)."'";

		// Insert/replace the record.
		$sql = "INSERT INTO `results` (`run`,`time`,`parameters`,`lnlikelihood`) VALUES ($run,NOW(),'$parameters',$lnlikelihood)";
		$result = mysql_query($sql);
		if (!$result) err("Unable to insert or update row in table. SQL statement: $sql");
	}

	function err($msg) {
		header("HTTP/1.1 500 Internal Server Error");
		echo $msg;
		exit();
	}

?>
