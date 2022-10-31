// Databricks notebook source
def mountFolder(dstStorageAccount:String, dstContainerName:String, dstFolderPath:String) {
  import java.lang.IllegalArgumentException
  
  val containerName     : String = dstContainerName
  val folderName        : String = dstFolderPath
  val storageAccount    : String = dstStorageAccount
  val appRegistrationId : String = "5da2fe00-7ce9-4be8-8f11-57befc2cf831"  // Application (client) ID
  val directoryId       : String = "628574bc-9c62-4216-afdd-7278cbd0123a"  // Directory (tenant) ID

  // Acquire secret from Azure Key Vault
//   val clientSecret = dbutils.secrets.get(scope = "azure-key-vault", key = "dev-raic-ingest-adlsgen2-001")
  // dev-raic--ingest-adlsgen2-001
  val clientSecret = "n9O7Q~EwVaFljyDlKV4VyXfYmIjK4qUW2osNp"
  // Uses defined properties from above
  val storageSource         : String = s"abfss://$containerName@$storageAccount.dfs.core.windows.net/$folderName"
  val fullMountLocationName : String = s"/mnt/$containerName/$folderName" 
  
  val configs = Map(
          "fs.azure.account.auth.type"              -> "OAuth"
        , "fs.azure.account.oauth.provider.type"    -> "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider"
        , "fs.azure.account.oauth2.client.id"       -> appRegistrationId
        , "fs.azure.account.oauth2.client.secret"   -> clientSecret
        , "fs.azure.account.oauth2.client.endpoint" -> s"https://login.microsoftonline.com/$directoryId/oauth2/token"
        )

  try {
        dbutils.fs.mount(
              source = storageSource
              , mountPoint = fullMountLocationName
              , extraConfigs = configs
              ) 
        println(s"Mount Location Created at: /mnt/$containerName/$folderName")
    
  } catch {
        case e: java.rmi.RemoteException => {
          println(s"Directory is Already Mounted at: /mnt/$containerName/$folderName")
        }
    
        case e: Exception => {
          println("There was some other error")
        }
  }
}

// COMMAND ----------

// mountFolder("devstraicscussynth", "maps-databricks", "mapdb")
// mountFolder("rebellion", "upload-00002", "CS-5343_map")

// COMMAND ----------

def listMounts() {
  // list out all mount locations
  display(dbutils.fs.mounts().toDF) 
}

// COMMAND ----------

def unMountFolder(dstFolderPath:String) {
    dbutils.fs.unmount(dstFolderPath)
}

// COMMAND ----------

listMounts()

// COMMAND ----------



// COMMAND ----------



// COMMAND ----------


